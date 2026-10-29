"""Does the recorded order flow predict future returns? A simple, hard-to-fool test to
run before giving the data to NEAT.

For each candle (GRANULARITY) with enough recorded minutes, it builds flow and
order-book values and checks how well each one ranks the next 1 and 4 candles' returns
(Spearman rank correlation, "IC"), across all coins. Then:
- t-stat: is the IC distinguishable from zero? (|t| > 3 is a reasonable bar here,
  since many values are tested and 4-candle returns overlap)
- stability: does the IC keep its sign in the first and second half of the data?
- spread: the return of the top 20% candles minus the bottom 20%, in bps, against the
  round-trip fee (2 x MAKER_FEE). Below that it can't pay for its own trades.

    docker run --rm -v /DATA/AppData/neatbase/data:/data ghcr.io/arandomenderman/neatbase:latest \\
        python -m app.flowcheck
"""
import argparse
import math
import os
import sqlite3
import time

import numpy as np

from . import coinbase_data, config
from .recorder import FLOW_DB

MIN_MINUTES = 50  # of 60 for an hourly candle; fewer = recorder was down, skip the candle


def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def spearman(a, b):
    if len(a) < 10:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def bars_from_flow(rows, g):
    """rows: (minute, buy_notional, sell_notional, n_buy, n_sell, imb10, imb50, imb100,
    micro, spread, missed). Returns {bar start: values} for bars with enough minutes."""
    acc = {}
    for m, bn, sn, nb, ns, i10, i50, i100, mp, sp, missed in rows:
        a = acc.setdefault(m // g * g, {"min": 0, "bn": 0.0, "sn": 0.0, "n": 0, "missed": 0, "book": []})
        a["min"] += 1
        a["bn"] += bn or 0.0
        a["sn"] += sn or 0.0
        a["n"] += (nb or 0) + (ns or 0)
        a["missed"] += missed or 0
        if i10 is not None:
            a["book"].append((i10, i50, i100, mp, sp))
    need = MIN_MINUTES * g / 3600
    return {t: a for t, a in acc.items() if a["min"] >= need and a["book"]}


def features(bars, g):
    """{bar: {name: value}} including multi-bar values that need the previous bars."""
    out = {}
    for t in sorted(bars):
        a = bars[t]
        book = np.array(a["book"])
        tot = a["bn"] + a["sn"]
        f = {"flow_imbalance_1": (a["bn"] - a["sn"]) / tot if tot else 0.0,
             "book_imb_10bps": book[:, 0].mean(), "book_imb_50bps": book[:, 1].mean(),
             "book_imb_100bps": book[:, 2].mean(), "microprice_bps": book[:, 3].mean(),
             "spread_bps": book[:, 4].mean()}
        prev = [bars[t - k * g] for k in range(4) if t - k * g in bars]
        bn, sn = sum(p["bn"] for p in prev), sum(p["sn"] for p in prev)
        f["flow_imbalance_4"] = (bn - sn) / (bn + sn) if bn + sn else 0.0
        hist = [bars[t - k * g]["n"] for k in range(1, 25) if t - k * g in bars]
        f["activity_vs_24"] = math.log((a["n"] + 1) / (np.mean(hist) + 1)) if hist else 0.0
        out[t] = f
    return out


def analyze(samples, fee_bps):
    """samples: list of (time, {feature: value}, next-candle log return, next-4 log return
    or nan). Returns one dict of statistics per feature."""
    names = list(samples[0][1])
    x = np.array([[f[k] for k in names] for _, f, _, _ in samples])
    y1 = np.array([s[2] for s in samples])
    y4 = np.array([s[3] for s in samples])
    t = np.array([s[0] for s in samples])
    n = len(samples)
    half = t < np.median(t)
    ok = ~np.isnan(y4)
    out = []
    for j, name in enumerate(names):
        ic = spearman(x[:, j], y1)
        tstat = ic * math.sqrt((n - 2) / max(1e-12, 1 - ic * ic))
        h1, h2 = spearman(x[half, j], y1[half]), spearman(x[~half, j], y1[~half])
        ic4 = spearman(x[ok, j], y4[ok])
        q20, q80 = np.quantile(x[ok, j], [0.2, 0.8])
        spread = (y4[ok][x[ok, j] >= q80].mean() - y4[ok][x[ok, j] <= q20].mean()) * 1e4
        out.append({"name": name, "ic": ic, "t": tstat, "ic_h1": h1, "ic_h2": h2, "ic4": ic4,
                    "spread_bps": spread, "beats_fee": abs(spread) > fee_bps,
                    "consistent": abs(tstat) > 3 and np.sign(h1) == np.sign(h2) == np.sign(ic)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=FLOW_DB)
    args = ap.parse_args()
    if not os.path.exists(args.db):
        raise SystemExit(f"No recorded data at {args.db} yet - is the recorder running?")
    g = config.GRANULARITY
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cache = coinbase_data.CandleCache(config.TRAINER_DB)
    samples = []
    print(f"Recorded data, {g // 60}m candles:")
    for pid in config.PRODUCT_IDS:
        rows = db.execute("SELECT minute, buy_notional, sell_notional, n_buy, n_sell, imb_10bps, imb_50bps, "
                          "imb_100bps, microprice_bps, spread_bps, missed_trades FROM flow WHERE product=? "
                          "ORDER BY minute", (pid,)).fetchall()
        if not rows:
            print(f"  {pid}: nothing recorded")
            continue
        bars = bars_from_flow(rows, g)
        missed = sum(r[10] or 0 for r in rows)
        trades = sum((r[3] or 0) + (r[4] or 0) for r in rows)
        print(f"  {pid}: {len(rows)} minutes over {(rows[-1][0] - rows[0][0]) / 86400:.1f} days, {len(bars)} "
              f"usable candles, {trades} trades ({100 * missed / max(1, trades + missed):.2f}% missed)")
        if not bars:
            continue
        candles = cache.get(pid, g, min(bars) - g, int(time.time()))
        close = {int(c[0]): c[4] for c in candles}
        for t, f in sorted(features(bars, g).items()):
            c0, c1, c4 = close.get(t), close.get(t + g), close.get(t + 4 * g)
            if c0 and c1:
                samples.append((t, f, math.log(c1 / c0), math.log(c4 / c0) if c4 else float("nan")))

    n = len(samples)
    if n < 50:
        raise SystemExit(f"\nOnly {n} usable candles with a known next return. Keep the recorder running; "
                         f"a first look needs ~500+ (about 1 week for 3 coins at 1h), a real answer ~3000+.")
    fee_bps = 2 * config.MAKER_FEE * 1e4
    print(f"\n{n} candles pooled across coins. Round-trip maker fee: {fee_bps:.0f} bps.\n")
    print(f"{'value':18} {'IC next':>8} {'t':>6} {'1st half':>9} {'2nd half':>9} {'IC next 4':>10} "
          f"{'top-bottom 20%, next 4':>23}")
    for r in analyze(samples, fee_bps):
        print(f"{r['name']:18} {r['ic']:+8.3f} {r['t']:+6.1f} {r['ic_h1']:+9.3f} {r['ic_h2']:+9.3f} "
              f"{r['ic4']:+10.3f} {r['spread_bps']:+15.0f} bps{' > fee' if r['beats_fee'] else ''}"
              f"{'  <- consistent' if r['consistent'] else ''}")
    print("\nHow to read it: an IC of 0.02-0.05 that holds in both halves is already a usable signal in "
          "markets; t > 3 with this many tests. If no value clears that, recording longer won't hurt, but "
          "there's no evidence yet. 'top-bottom 20%' must beat the fee to pay for trading on it alone.")


if __name__ == "__main__":
    main()
