"""Honest out-of-sample comparison of training settings (walk-forward).

The last LOCKBOX days are kept out of every development run. Before that, FOLDS test
months are laid end to end going back in time; for each one, fresh populations are
trained on the TRAIN_DAYS right before it and scored on it. Nothing is ever picked
using test data, so the per-month results are honest, and averaging over several
months (bull, bear, sideways) keeps one lucky month from deciding.

Iterate on settings with the default (development) mode. Run --final only once you're
done: it trains on the TRAIN_DAYS before the lockbox and scores the lockbox. If that
looks much worse than development, the settings were tuned to the development months.

    docker run --rm -v neatbase-compare:/data ghcr.io/arandomenderman/neatbase:latest \\
        python -m app.compare --granularities 3600,21600 --folds 6 --runs 2

Every bot setting applies (pass with -e, e.g. -e MAX_POSITION_FRACTION=1.0).
"""
import argparse
import logging
import os
import statistics
import time

import neat

from . import coinbase_data, config, fees, trainer
from .fastsim import evaluate


def train_population(cfg, train_w, costs, generations, workers):
    pop = neat.Population(cfg)
    with trainer.BatchEvaluator(workers, train_w, costs, cfg) as pe:
        pop.run(pe.evaluate, generations)
    return pop.best_genome


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--granularities", default="3600",
                    help="candle sizes in seconds, comma separated (3600=1h, 7200=2h, 21600=6h, 86400=1d)")
    ap.add_argument("--folds", type=int, default=6, help="test months, laid end to end back in time")
    ap.add_argument("--fold-days", type=float, default=30)
    ap.add_argument("--lockbox-days", type=float, default=30, help="most recent days, only used by --final")
    ap.add_argument("--final", action="store_true", help="score the lockbox instead (do this once, at the end)")
    ap.add_argument("--runs", type=int, default=2, help="independent training runs per test month")
    ap.add_argument("--ensemble", type=int, default=1, help="independent populations voting per run")
    ap.add_argument("--generations", type=int, default=config.GENERATIONS_PER_CYCLE)
    ap.add_argument("--pop", type=int, default=config.POP_SIZE)
    ap.add_argument("--train-days", type=float, default=config.TRAIN_DAYS)
    ap.add_argument("--workers", type=int, default=0, help="CPU cores to use, 0 = all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("app.coinbase_data").setLevel(logging.INFO)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    cfg = trainer.load_neat_config()
    cfg.pop_size = args.pop
    cache = coinbase_data.CandleCache(config.TRAINER_DB)
    workers = args.workers or os.cpu_count() or 1
    taker, maker = fees.resolve_fees()
    grans = [int(g) for g in args.granularities.split(",")]
    for g in grans:
        if g not in config.GRANULARITY_NAMES:
            raise SystemExit(f"{g} is not a Coinbase candle size: {sorted(config.GRANULARITY_NAMES)}")

    day = 86400
    now = int(time.time())
    end = now - now % day
    lock = end - int(args.lockbox_days * day)
    if args.final:
        tests = [(lock, end)]
        label = f"FINAL: lockbox = last {args.lockbox_days:g} days"
    else:
        fd = int(args.fold_days * day)
        tests = [(lock - (k + 1) * fd, lock - k * fd) for k in range(args.folds)][::-1]
        label = (f"development: {args.folds} test months of {args.fold_days:g}d before the "
                 f"{args.lockbox_days:g}d lockbox")
    print(f"{', '.join(config.PRODUCT_IDS)} | {label}\n{args.runs} runs x ensemble of {args.ensemble} x "
          f"{args.generations} generations x {args.pop} genomes, trained on the {args.train_days:g}d before each "
          f"test | {config.ORDER_TYPE} orders, max {config.MAX_POSITION_FRACTION:.0%} invested, scoring "
          f"{config.FITNESS_MODE} | {workers} cores\n", flush=True)

    summary = []
    for g in grans:
        config.GRANULARITY = g
        first_train = tests[0][0] - int(args.train_days * day)
        market = trainer.load_market(cache, first_train, tests[-1][1])
        costs = trainer.make_costs(taker, maker)
        alphas, rets, bhs, month_means = [], [], [], []
        t_all = time.time()
        for t0, t1 in tests:
            train_w = trainer.windows(market, t0 - int(args.train_days * day), t0)
            test_w = trainer.windows(market, t0, t1)
            month = []
            for _ in range(args.runs):
                members = [train_population(cfg, train_w, costs, args.generations, workers)
                           for _ in range(args.ensemble)]
                tr = evaluate(members, cfg, train_w, costs, chunks=config.FITNESS_CHUNKS)
                te = evaluate(members, cfg, test_w, costs)
                month.append(te["alpha_pct"])
                alphas.append(te["alpha_pct"])
                rets.append(te["return_pct"])
                print(f"{g // 60:>5}m  {time.strftime('%Y-%m-%d', time.gmtime(t0))}: train alpha "
                      f"{tr['alpha_pct']:+6.2f} | unseen alpha {te['alpha_pct']:+6.2f}, return "
                      f"{te['return_pct']:+6.2f}% ({te['avg_exposure']:.0%} invested, buy&hold "
                      f"{te['buy_hold_pct']:+6.2f}%, {te['trades']} trades, fees ${te['fees_paid']:.2f})",
                      flush=True)
            bhs.append(te["buy_hold_pct"])
            month_means.append(sum(month) / len(month))
        summary.append((g, alphas, rets, bhs, month_means, time.time() - t_all))
        print()

    print("Unseen alpha per test month (return beyond holding the same average % invested; "
          "averaged over runs; log-%):")
    print(f"{'candle':>7} {'mean':>7} {'median':>7} {'worst mo':>9} {'best mo':>8} {'months>0':>9} "
          f"{'mean ret':>9} {'buy&hold':>9} {'time':>6}")
    for g, alphas, rets, bhs, mm, dt in summary:
        print(f"{g // 60:>6}m {sum(alphas) / len(alphas):+7.2f} {statistics.median(alphas):+7.2f} "
              f"{min(mm):+9.2f} {max(mm):+8.2f} {sum(m > 0 for m in mm):>4}/{len(mm):<4} "
              f"{sum(rets) / len(rets):+8.2f}% {sum(bhs) / len(bhs):+8.2f}% {dt:5.0f}s")
    print("\nReal skill = mean alpha clearly above 0 AND most months above 0. Fees alone put a "
          "do-little bot slightly below 0.")


if __name__ == "__main__":
    main()
