"""Compare candle sizes (or any setting that changes training) on unseen data.

For each granularity, train RUNS fresh populations on the same TRAIN_DAYS, then score
each run's best network on the following VAL_DAYS, which training never saw. Nothing is
picked using the validation data, so the numbers are an honest out-of-sample estimate.

    docker run --rm -v neatbase-compare:/data ghcr.io/arandomenderman/neatbase:latest \\
        python -m app.compare --granularities 3600,7200,21600

Uses the same fees, order rules and position sizing as the bot (all settings apply).
"""
import argparse
import logging
import os
import statistics
import time

import neat

from . import coinbase_data, config, fees, trainer
from .sim import backtest_many


def run_one(cfg, train_w, val_w, costs, generations, workers):
    pop = neat.Population(cfg)
    with neat.ParallelEvaluator(workers, trainer._eval_genome, initializer=trainer._init_worker,
                                initargs=((train_w, costs),)) as pe:
        pop.run(pe.evaluate, generations)
    best = pop.best_genome
    net = neat.nn.FeedForwardNetwork.create(best, cfg)
    return (backtest_many(net.activate, train_w, costs, chunks=config.FITNESS_CHUNKS),
            backtest_many(net.activate, val_w, costs), best.size())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--granularities", default="3600,7200,21600",
                    help="candle sizes in seconds, comma separated (3600=1h, 7200=2h, 21600=6h, 86400=1d)")
    ap.add_argument("--runs", type=int, default=3, help="independent training runs per candle size")
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop", type=int, default=config.POP_SIZE)
    ap.add_argument("--train-days", type=float, default=90)
    ap.add_argument("--val-days", type=float, default=30)
    ap.add_argument("--workers", type=int, default=0, help="CPU cores to use, 0 = all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    cfg = trainer.load_neat_config()
    cfg.pop_size = args.pop
    cache = coinbase_data.CandleCache(config.TRAINER_DB)
    workers = args.workers or os.cpu_count() or 1
    taker, maker = fees.resolve_fees()
    grans = [int(g) for g in args.granularities.split(",")]
    print(f"{', '.join(config.PRODUCT_IDS)} | {args.runs} runs x {args.generations} generations x "
          f"{args.pop} genomes | train {args.train_days:g}d, test {args.val_days:g}d unseen | "
          f"{config.ORDER_TYPE} orders, max {config.MAX_POSITION_FRACTION:.0%} invested | {workers} cores\n")

    now = int(time.time())
    end = now - now % max(grans)  # same test window for every candle size
    val_start = end - int(args.val_days * 86400)
    train_start = val_start - int(args.train_days * 86400)
    rows = []
    for g in grans:
        if g not in config.GRANULARITY_NAMES:
            raise SystemExit(f"{g} is not a Coinbase candle size: {sorted(config.GRANULARITY_NAMES)}")
        config.GRANULARITY = g
        market = trainer.load_market(cache, train_start, end)
        costs = trainer.make_costs(taker, maker)
        train_w = trainer.windows(market, train_start, val_start)
        val_w = trainer.windows(market, val_start, end)
        for r in range(args.runs):
            t0 = time.time()
            tr, va, (nodes, conns) = run_one(cfg, train_w, val_w, costs, args.generations, workers)
            rows.append((g, tr, va))
            print(f"{g // 60:>5}m run {r + 1}: train {tr['return_pct']:+7.2f}% alpha {tr['alpha_pct']:+6.2f} | "
                  f"unseen {va['return_pct']:+7.2f}% alpha {va['alpha_pct']:+6.2f} "
                  f"({va['avg_exposure']:.0%} invested, buy&hold {va['buy_hold_pct']:+.2f}%) | "
                  f"{va['trades']:3d} trades | fees ${va['fees_paid']:7.2f} | {nodes}n/{conns}c | "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"\nSummary on unseen data (median of runs; per-coin pot $1000; buy & hold is 100% invested)."
          f"\nalpha = return beyond holding the same average % invested the whole time "
          f"(scoring: {config.FITNESS_MODE}):")
    print(f"{'candle':>8} {'train α':>9} {'unseen':>9} {'unseen α':>9} {'worst α':>9} {'invested':>9} "
          f"{'buy&hold':>9} {'trades':>7} {'max DD':>7}")
    for g in grans:
        rs = [(tr, va) for gg, tr, va in rows if gg == g]
        med = lambda xs: statistics.median(xs)
        print(f"{g // 60:>7}m {med([t['alpha_pct'] for t, _ in rs]):+9.2f} "
              f"{med([v['return_pct'] for _, v in rs]):+8.2f}% {med([v['alpha_pct'] for _, v in rs]):+9.2f} "
              f"{min(v['alpha_pct'] for _, v in rs):+9.2f} {med([v['avg_exposure'] for _, v in rs]):9.0%} "
              f"{rs[0][1]['buy_hold_pct']:+8.2f}% {med([v['trades'] for _, v in rs]):7.0f} "
              f"{med([v['max_drawdown_pct'] for _, v in rs]):6.2f}%")
    print("\nReal timing skill = positive 'unseen α' in every run ('worst α' > 0). Alpha near zero or below "
          "means it's no better than just holding that much.")


if __name__ == "__main__":
    main()
