"""Evolves NEAT trading networks on recent Coinbase candles and promotes a champion
for the trader to use. Runs forever: train -> validate -> maybe promote -> sleep."""
import json
import logging
import os
import pickle
import sqlite3
import time

import neat

from . import coinbase_data, config, features, fees
from .sim import Costs, backtest

log = logging.getLogger("trainer")

_DATA = None  # (feats, candles, start, end, costs), set in each worker process


def _init_worker(data):
    global _DATA
    _DATA = data


def _eval_genome(genome, cfg):
    net = neat.nn.FeedForwardNetwork.create(genome, cfg)
    feats, candles, start, end, costs = _DATA
    return backtest(net.activate, feats, candles, start, end, costs,
                    chunks=config.FITNESS_CHUNKS)["fitness"]


def load_neat_config():
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                      neat.DefaultStagnation, config.NEAT_CONFIG_PATH)
    cfg.pop_size = config.POP_SIZE
    assert cfg.genome_config.num_inputs == features.N_INPUTS
    return cfg


def load_champion():
    try:
        with open(config.CHAMPION_PATH, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None


def save_champion(genome, meta):
    tmp = config.CHAMPION_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"genome": genome, "meta": meta}, f)
    os.replace(tmp, config.CHAMPION_PATH)


class DBReporter(neat.reporting.BaseReporter):
    def __init__(self, db):
        self.db = db
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, cfg, population, species, best_genome):
        fits = [g.fitness for g in population.values()]
        self.db.execute("INSERT INTO generations VALUES (?,?,?,?,?)",
                        (time.time(), self.generation, best_genome.fitness,
                         sum(fits) / len(fits), len(species.species)))
        self.db.commit()
        log.info("gen %d  best %.3f  mean %.3f  species %d", self.generation,
                 best_genome.fitness, sum(fits) / len(fits), len(species.species))


def open_db():
    db = sqlite3.connect(config.TRAINER_DB, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS generations (
        ts REAL, generation INTEGER, best REAL, mean REAL, species INTEGER)""")
    db.execute("""CREATE TABLE IF NOT EXISTS runs (
        ts REAL, generation INTEGER, train TEXT, validation TEXT,
        champion_validation TEXT, promoted INTEGER, nodes INTEGER, conns INTEGER,
        costs TEXT, val_curve TEXT)""")
    db.commit()
    return db


def restore_population(cfg):
    sig = {"inputs": features.N_INPUTS, "pop": config.POP_SIZE}
    sig_path = config.CHECKPOINT_PATH + ".json"
    try:
        with open(sig_path) as f:
            if json.load(f) == sig:
                pop = neat.Checkpointer.restore_checkpoint(config.CHECKPOINT_PATH, new_config=cfg)
                log.info("Resumed population at generation %d", pop.generation)
                return pop
    except Exception as e:
        if os.path.exists(config.CHECKPOINT_PATH):
            log.warning("Could not restore checkpoint (%s); starting fresh", e)
    with open(sig_path, "w") as f:
        json.dump(sig, f)
    return neat.Population(cfg)


def save_population(pop, cfg):
    tmp = config.CHECKPOINT_PATH + ".tmp"
    reporters, pop.species.reporters = pop.species.reporters, None  # holds an unpicklable DB handle
    try:
        neat.Checkpointer(None, filename_prefix=tmp).save_checkpoint(
            cfg, pop.population, pop.species, pop.generation)
    finally:
        pop.species.reporters = reporters
    os.replace(f"{tmp}{pop.generation}", config.CHECKPOINT_PATH)


def _downsample(curve, n=400):
    step = max(1, len(curve) // n)
    return [[int(t), round(e, 2)] for t, e in curve[::step]]


def train_cycle(pop, cfg, cache, db, taker_fee):
    g = config.GRANULARITY
    now = int(time.time())
    span = (config.TRAIN_DAYS + config.VALIDATION_DAYS) * 86400 + features.WARMUP * g
    candles = cache.get(config.PRODUCT_ID, g, now - int(span), now)
    val_bars = int(config.VALIDATION_DAYS * 86400 / g)
    if len(candles) < features.WARMUP + val_bars + 100:
        raise RuntimeError(f"Only {len(candles)} candles available; need more history")
    feats = features.market_features(candles, g)
    split = len(candles) - val_bars

    try:
        live_spread = coinbase_data.current_spread_bps(config.PRODUCT_ID)
    except Exception:
        live_spread = 0.0
    costs = Costs(taker_fee=taker_fee, slippage_bps=config.SLIPPAGE_BPS,
                  spread_bps=max(config.SPREAD_BPS, live_spread))
    log.info("Training on %d bars, validating on %d bars | fee %.3f%% slip %.1fbps spread %.2fbps",
             split - features.WARMUP, val_bars, costs.taker_fee * 100, costs.slippage_bps,
             costs.spread_bps)

    workers = config.NUM_WORKERS or os.cpu_count() or 1
    data = (feats, candles, features.WARMUP, split, costs)
    pop.best_genome = None  # only compare genomes scored on this cycle's data
    with neat.ParallelEvaluator(workers, _eval_genome, initializer=_init_worker,
                                initargs=(data,)) as pe:
        pop.run(pe.evaluate, config.GENERATIONS_PER_CYCLE)
    save_population(pop, cfg)

    def score(genome, a, b, **kw):
        net = neat.nn.FeedForwardNetwork.create(genome, cfg)
        return backtest(net.activate, feats, candles, a, b, costs, **kw)

    cand = pop.best_genome
    train = score(cand, features.WARMUP, split, chunks=config.FITNESS_CHUNKS)
    val = score(cand, split, len(candles), record=True)
    champ = load_champion()
    champ_val = score(champ["genome"], split, len(candles)) if champ else None
    promote = champ is None or val["fitness"] > champ_val["fitness"]

    nodes, conns = cand.size()
    curve = val.pop("curve")
    val.pop("log")
    log.info("Candidate: train %+.2f%% | validation %+.2f%% (buy&hold %+.2f%%, %d trades, "
             "fees $%.2f) | champion validation %s -> %s",
             train["return_pct"], val["return_pct"], val["buy_hold_pct"], val["trades"],
             val["fees_paid"], f"{champ_val['return_pct']:+.2f}%" if champ_val else "none",
             "PROMOTED" if promote else "kept old champion")
    if promote:
        save_champion(cand, {
            "promoted_at": time.time(), "generation": pop.generation, "train": train,
            "validation": val, "nodes": nodes, "conns": conns,
            "product": config.PRODUCT_ID, "granularity": g,
        })
    db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)", (
        time.time(), pop.generation, json.dumps(train), json.dumps(val),
        json.dumps(champ_val) if champ_val else None, int(promote), nodes, conns,
        json.dumps(costs.__dict__), json.dumps(_downsample(curve))))
    db.commit()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    cfg = load_neat_config()
    pop = restore_population(cfg)
    db = open_db()
    pop.add_reporter(DBReporter(db))
    cache = coinbase_data.CandleCache(config.TRAINER_DB)
    taker_fee = fees.resolve_taker_fee()
    while True:
        try:
            train_cycle(pop, cfg, cache, db, taker_fee)
        except Exception:
            log.exception("Training cycle failed; retrying after the interval")
        log.info("Next training cycle in %.1f hours", config.RETRAIN_INTERVAL_HOURS)
        time.sleep(config.RETRAIN_INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    main()
