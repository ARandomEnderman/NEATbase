"""Evolves NEAT trading networks on recent Coinbase candles and promotes a champion for
the trader. Runs forever, every RETRAIN_INTERVAL_HOURS:

1. evolve on the training window (all coins, one shared network)
2. the cycle's best network must beat the champion on the held-out validation window to
   become a *challenger* (at most one new challenger per day: the best of that day)
3. a challenger is promoted only after it also beats the champion on PROBATION_DAYS of
   data that didn't exist when it was created. Each challenger gets one forward test.
"""
import json
import logging
import os
import pickle
import sqlite3
import time
import uuid
from itertools import count

import neat

from . import coinbase_data, config, features, fees
from .sim import Costs, backtest_many, window

log = logging.getLogger("trainer")

_DATA = None  # (windows, costs), set in each worker process


def _init_worker(data):
    global _DATA
    _DATA = data


def _eval_genome(genome, cfg):
    net = neat.nn.FeedForwardNetwork.create(genome, cfg)
    windows, costs = _DATA
    return backtest_many(net.activate, windows, costs, chunks=config.FITNESS_CHUNKS)["fitness"]


def load_neat_config():
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                      neat.DefaultStagnation, config.NEAT_CONFIG_PATH)
    cfg.pop_size = config.POP_SIZE
    assert cfg.genome_config.num_inputs == features.N_INPUTS
    return cfg


def _load(path):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError):
        return None
    if obj["meta"].get("n_inputs") != features.N_INPUTS:
        return None  # trained for a different set of inputs
    return obj


def _save(path, genome, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"genome": genome, "meta": {**meta, "n_inputs": features.N_INPUTS}}, f)
    os.replace(tmp, path)


def load_champion():
    return _load(config.CHAMPION_PATH)


def save_champion(genome, meta):
    _save(config.CHAMPION_PATH, genome, meta)


def _challenger_path(cid):
    return os.path.join(config.CHALLENGERS_DIR, f"{cid}.pkl")


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
    db.execute("""CREATE TABLE IF NOT EXISTS challengers (
        id TEXT PRIMARY KEY, born REAL, generation INTEGER, nodes INTEGER, conns INTEGER,
        validation TEXT, status TEXT, forward TEXT, champion_forward TEXT, decided REAL)""")
    cols = [r[1] for r in db.execute("PRAGMA table_info(runs)")]
    if "status" not in cols:
        db.execute("ALTER TABLE runs ADD COLUMN status TEXT")
    db.commit()
    return db


def restore_population(cfg):
    sig = {"inputs": features.N_INPUTS, "pop": config.POP_SIZE, "products": config.PRODUCT_IDS}
    sig_path = config.CHECKPOINT_PATH + ".json"
    try:
        with open(sig_path) as f:
            if json.load(f) == sig:
                pop = neat.Checkpointer.restore_checkpoint(config.CHECKPOINT_PATH, new_config=cfg)
                # neat-python would otherwise restart new-node ids from whichever genome
                # mutates first, colliding with node ids other genomes already use.
                top = max(k for g in pop.population.values() for k in g.nodes)
                cfg.genome_config.node_indexer = count(top + 1)
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


def _strip(result):
    return {k: v for k, v in result.items() if k != "curve"}


def load_market(cache, t0, t1):
    """{product: (feats, candles)} for candles starting in [t0 - warmup, t1)."""
    g = config.GRANULARITY
    out = {}
    for pid in config.PRODUCT_IDS:
        candles = cache.get(pid, g, int(t0 - features.WARMUP * g), int(t1))
        if len(candles) > features.WARMUP + 100:
            out[pid] = (features.market_features(candles, g), candles)
        else:
            log.warning("Skipping %s: only %d candles", pid, len(candles))
    if not out:
        raise RuntimeError("No product has enough candle history")
    return out


def windows(market, t0, t1):
    return {pid: window(f, c, t0, t1) for pid, (f, c) in market.items()}


def make_costs(taker, maker):
    spreads = []
    for pid in config.PRODUCT_IDS:
        try:
            spreads.append(coinbase_data.current_spread_bps(pid))
        except Exception:
            pass
    spread = max([config.SPREAD_BPS] + spreads)
    return Costs(taker_fee=taker, maker_fee=maker, slippage_bps=config.SLIPPAGE_BPS, spread_bps=spread)


def add_challenger(db, cand, now_bar, generation, val):
    """Keep at most one challenger per day: the best one by validation fitness."""
    recent = db.execute("SELECT id, validation FROM challengers WHERE status='waiting' "
                        "AND born > ? AND born <= ?", (now_bar - 86400, now_bar)).fetchone()
    if recent:
        if json.loads(recent[1])["fitness"] >= val["fitness"]:
            return "passed validation; today's challenger is better"
        db.execute("UPDATE challengers SET status='replaced', decided=? WHERE id=?", (time.time(), recent[0]))
        _remove(recent[0])
    waiting = db.execute("SELECT COUNT(*) FROM challengers WHERE status='waiting'").fetchone()[0]
    if waiting >= config.MAX_CHALLENGERS:
        return "passed validation; probation queue full"
    cid = uuid.uuid4().hex[:10]
    nodes, conns = cand.size()
    _save(_challenger_path(cid), cand, {"born": now_bar, "generation": generation})
    db.execute("INSERT INTO challengers VALUES (?,?,?,?,?,?,?,?,?,?)", (
        cid, now_bar, generation, nodes, conns, json.dumps(val), "waiting", None, None, None))
    return "challenger: on probation"


def _remove(cid):
    try:
        os.remove(_challenger_path(cid))
    except FileNotFoundError:
        pass


def run_probations(db, cfg, market, costs, now_bar):
    """Forward-test every challenger whose probation period is over, against the
    champion, on data newer than the challenger. Promote the best winner."""
    champ = load_champion()
    due = db.execute("SELECT id, born, generation, nodes, conns FROM challengers "
                     "WHERE status='waiting' AND born <= ?",
                     (now_bar - config.PROBATION_DAYS * 86400,)).fetchall()
    best = None
    for cid, born, generation, nodes, conns in due:
        ch = _load(_challenger_path(cid))
        if ch is None or champ is None:
            db.execute("UPDATE challengers SET status='dropped', decided=? WHERE id=?", (time.time(), cid))
            continue
        win = windows(market, born, now_bar)
        fwd = backtest_many(neat.nn.FeedForwardNetwork.create(ch["genome"], cfg).activate, win, costs)
        cfwd = backtest_many(neat.nn.FeedForwardNetwork.create(champ["genome"], cfg).activate, win, costs)
        won = fwd["fitness"] > cfwd["fitness"]
        log.info("Probation %s: forward %+.2f%% vs champion %+.2f%% (%d vs %d trades) -> %s", cid,
                 fwd["return_pct"], cfwd["return_pct"], fwd["trades"], cfwd["trades"],
                 "WON" if won else "lost")
        db.execute("UPDATE challengers SET status=?, forward=?, champion_forward=?, decided=? WHERE id=?",
                   ("won" if won else "lost", json.dumps(fwd), json.dumps(cfwd), time.time(), cid))
        if won and (best is None or fwd["fitness"] > best[1]["fitness"]):
            best = (cid, fwd, ch, generation, nodes, conns)
    if best:
        cid, fwd, ch, generation, nodes, conns = best
        save_champion(ch["genome"], {
            "promoted_at": time.time(), "generation": generation, "nodes": nodes, "conns": conns,
            "how": f"won {config.PROBATION_DAYS:g}-day forward test", "challenger": cid,
            "forward": _strip(fwd), "products": config.PRODUCT_IDS,
            "granularity": config.GRANULARITY})
        db.execute("UPDATE challengers SET status='promoted' WHERE id=?", (cid,))
        log.info("Promoted challenger %s to champion", cid)
    for (cid, *_rest) in due:
        _remove(cid)
    db.commit()


def train_cycle(pop, cfg, cache, db, fee_rates):
    g = config.GRANULARITY
    now = int(time.time())
    now_bar = now - now % g
    val_start = now_bar - int(config.VALIDATION_DAYS * 86400)
    train_start = val_start - int(config.TRAIN_DAYS * 86400)
    market = load_market(cache, train_start, now_bar)
    costs = make_costs(*fee_rates)
    train_w, val_w = windows(market, train_start, val_start), windows(market, val_start, now_bar)
    log.info("Training on %s: %d days, validating on %d days | %s orders, maker %.2f%% taker %.2f%% "
             "slip %.1fbps spread %.2fbps", ",".join(market), config.TRAIN_DAYS, config.VALIDATION_DAYS,
             costs.order_type, costs.maker_fee * 100, costs.taker_fee * 100, costs.slippage_bps,
             costs.spread_bps)

    workers = config.NUM_WORKERS or os.cpu_count() or 1
    pop.best_genome = None  # only compare genomes scored on this cycle's data
    with neat.ParallelEvaluator(workers, _eval_genome, initializer=_init_worker,
                                initargs=((train_w, costs),)) as pe:
        pop.run(pe.evaluate, config.GENERATIONS_PER_CYCLE)
    save_population(pop, cfg)

    def score(genome, win, **kw):
        return backtest_many(neat.nn.FeedForwardNetwork.create(genome, cfg).activate, win, costs, **kw)

    cand = pop.best_genome
    train = score(cand, train_w, chunks=config.FITNESS_CHUNKS)
    val = score(cand, val_w, record=True)
    champ = load_champion()
    champ_val = score(champ["genome"], val_w) if champ else None
    nodes, conns = cand.size()

    if champ is None:
        status = "promoted: first champion (no forward test yet)"
        save_champion(cand, {"promoted_at": time.time(), "generation": pop.generation, "nodes": nodes,
                             "conns": conns, "how": "first champion", "validation": _strip(val),
                             "products": list(market), "granularity": g})
    elif val["fitness"] > champ_val["fitness"]:
        status = add_challenger(db, cand, now_bar, pop.generation, _strip(val))
    else:
        status = "rejected: lost to champion on validation"
    log.info("Candidate: train %+.2f%% (alpha %+.2f) | validation %+.2f%% (alpha %+.2f, %.0f%% invested, "
             "buy&hold %+.2f%%, %d trades, fees $%.2f) | champion validation %s -> %s", train["return_pct"],
             train["alpha_pct"], val["return_pct"], val["alpha_pct"], 100 * val["avg_exposure"],
             val["buy_hold_pct"], val["trades"], val["fees_paid"],
             f"{champ_val['return_pct']:+.2f}%" if champ_val else "none", status)
    db.execute("INSERT INTO runs (ts, generation, train, validation, champion_validation, promoted, nodes, "
               "conns, costs, val_curve, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                   time.time(), pop.generation, json.dumps(_strip(train)), json.dumps(_strip(val)),
                   json.dumps(champ_val) if champ_val else None, int(status.startswith("promoted")),
                   nodes, conns, json.dumps(costs.__dict__), json.dumps(_downsample(val["curve"])), status))
    db.commit()
    run_probations(db, cfg, market, costs, now_bar)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.nice(10)  # training yields the CPU to everything else on the server
    cfg = load_neat_config()
    pop = restore_population(cfg)
    db = open_db()
    pop.add_reporter(DBReporter(db))
    cache = coinbase_data.CandleCache(config.TRAINER_DB)
    fee_rates = fees.resolve_fees()
    while True:
        try:
            train_cycle(pop, cfg, cache, db, fee_rates)
        except Exception:
            log.exception("Training cycle failed; retrying after the interval")
        log.info("Next training cycle in %.1f hours", config.RETRAIN_INTERVAL_HOURS)
        time.sleep(config.RETRAIN_INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    main()
