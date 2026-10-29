"""Fee-aware trading simulator shared by training (backtests) and paper trading.

Order types:
- taker: market order, fills at once, pays the taker fee plus spread and slippage.
- maker: post-only limit order at the best bid (buy) / best ask (sell) placed at a bar's
  close. It only fills if the *next* bar trades through the limit price (touching it is
  not enough), and pays the maker fee. Unfilled orders are cancelled. A sell that fails
  to fill MAKER_SELL_FALLBACK_BARS bars in a row becomes a market sell.

Equity is always marked at *liquidation value*: what a market sell would get right now,
after spread, slippage and the taker fee.
"""
import bisect
import math
from dataclasses import dataclass

from . import config
from .features import WARMUP, position_features


@dataclass
class Costs:
    taker_fee: float = config.TAKER_FEE
    maker_fee: float = config.MAKER_FEE
    slippage_bps: float = config.SLIPPAGE_BPS
    spread_bps: float = config.SPREAD_BPS
    order_type: str = config.ORDER_TYPE
    fallback_bars: int = config.MAKER_SELL_FALLBACK_BARS

    def buy_price(self, mid):
        return mid * (1 + self.spread_bps / 20_000 + self.slippage_bps / 10_000)

    def sell_price(self, mid):
        return mid * (1 - self.spread_bps / 20_000 - self.slippage_bps / 10_000)

    def bid(self, mid):
        return mid * (1 - self.spread_bps / 20_000)

    def ask(self, mid):
        return mid * (1 + self.spread_bps / 20_000)


def buy_with_cash(cash, price, fee_rate):
    """Spend `cash` in total (fee included). Returns (base_qty, notional, fee)."""
    notional = cash / (1 + fee_rate)
    return notional / price, notional, notional * fee_rate


def sell_base(qty, price, fee_rate):
    """Sell `qty`. Returns (cash_received, notional, fee)."""
    notional = qty * price
    fee = notional * fee_rate
    return notional - fee, notional, fee


def limit_fills(side, price, candle):
    """Did a resting limit order at `price` fill during `candle`? Requires a trade through."""
    return candle[3] < price if side == "buy" else candle[2] > price


def entry_fraction(output, max_fraction=config.MAX_POSITION_FRACTION, steps=config.SIZE_STEPS):
    """Share of the pot to invest on a buy: stronger signal -> bigger position, in steps."""
    strength = (output - config.BUY_THRESHOLD) / (1 - config.BUY_THRESHOLD)
    return max_fraction * min(steps, max(1, math.ceil(strength * steps))) / steps


def exposure(cash, position_value, max_fraction=config.MAX_POSITION_FRACTION):
    total = cash + position_value
    return position_value / total / max_fraction if total > 0 else 0.0


def decide(output, in_position):
    if not in_position and output > config.BUY_THRESHOLD:
        return "buy"
    if in_position and output < config.SELL_THRESHOLD:
        return "sell"
    return None


def backtest(activate, feats, candles, start, end, costs, start_cash=1000.0,
             max_fraction=config.MAX_POSITION_FRACTION, chunks=1, record=False):
    """Run a policy over bars [start, end). Decisions use bar i's closed candle."""
    maker = costs.order_type == "maker"
    cash, qty, entry_cost, held, prev_out = start_cash, 0.0, 0.0, 0, 0.0
    pending, sell_fails = None, 0
    trades, wins, fees = 0, 0, 0.0
    peak, max_dd = start_cash, 0.0
    bounds = set(start + (end - start) * k // chunks for k in range(1, chunks + 1))
    chunk_equity = [start_cash]
    curve, log = [], []

    def do_buy(spend, price, fee_rate, t):
        nonlocal cash, qty, entry_cost, held, fees
        q, _, fee = buy_with_cash(spend, price, fee_rate)
        cash -= spend
        qty, entry_cost, held, fees = q, spend, 0, fees + fee
        if record:
            log.append({"time": t, "side": "buy", "price": price, "fee": fee})

    def do_sell(price, fee_rate, t):
        nonlocal cash, qty, trades, wins, fees, sell_fails
        got, _, fee = sell_base(qty, price, fee_rate)
        cash += got
        trades += 1
        wins += got > entry_cost
        fees += fee
        if record:
            log.append({"time": t, "side": "sell", "price": price, "fee": fee, "pnl": got - entry_cost})
        qty, sell_fails = 0.0, 0

    for i in range(start, end):
        bar = candles[i]
        if pending:  # placed at the previous close, live during this bar
            side, price, amount = pending
            pending = None
            if limit_fills(side, price, bar):
                if side == "buy":
                    do_buy(amount, price, costs.maker_fee, bar[0])
                else:
                    do_sell(price, costs.maker_fee, bar[0])
            elif side == "sell":
                sell_fails += 1

        close = bar[4]
        sell_px = costs.sell_price(close)
        in_pos = qty > 0
        value = sell_base(qty, sell_px, costs.taker_fee)[0] if in_pos else 0.0
        pnl = value / entry_cost - 1 if in_pos else 0.0
        out = activate(list(feats[i]) + position_features(exposure(cash, value, max_fraction), pnl, held,
                                                          prev_out))[0]
        prev_out = out
        action = decide(out, in_pos)
        if action != "sell":
            sell_fails = 0

        spend = cash * entry_fraction(out, max_fraction) if action == "buy" else 0.0
        if action == "buy" and spend >= 1.0:
            if maker:
                pending = ("buy", costs.bid(close), spend)
            else:
                do_buy(spend, costs.buy_price(close), costs.taker_fee, bar[0])
        elif action == "sell":
            if maker and sell_fails < costs.fallback_bars:
                pending = ("sell", costs.ask(close), qty)
            else:
                do_sell(sell_px, costs.taker_fee, bar[0])
        elif qty > 0:
            held += 1

        equity = cash + (sell_base(qty, sell_px, costs.taker_fee)[0] if qty > 0 else 0.0)
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
        if i + 1 in bounds:
            chunk_equity.append(equity)
        if record:
            curve.append((bar[0], equity))

    if qty > 0:
        trades += 1  # an open position counts as a (marked-to-market) trade

    rets = [100 * math.log(max(b, 1e-9) / a) for a, b in zip(chunk_equity, chunk_equity[1:])]
    total = sum(rets)
    std = 0.0
    if len(rets) > 1:
        m = total / len(rets)
        std = math.sqrt(sum((r - m) ** 2 for r in rets) / len(rets))
    fitness = total - std - config.DRAWDOWN_PENALTY * 100 * max_dd
    if trades == 0:
        fitness -= config.INACTIVITY_PENALTY

    first, last = candles[start][4], candles[end - 1][4]
    bh = sell_base(buy_with_cash(start_cash, costs.buy_price(first), costs.taker_fee)[0],
                   costs.sell_price(last), costs.taker_fee)[0]
    result = {
        "fitness": fitness,
        "final_equity": chunk_equity[-1],
        "return_pct": 100 * (chunk_equity[-1] / start_cash - 1),
        "buy_hold_pct": 100 * (bh / start_cash - 1),
        "max_drawdown_pct": 100 * max_dd,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "fees_paid": fees,
    }
    if record:
        result["curve"], result["log"] = curve, log
    return result


def window(feats, candles, t0, t1):
    """(feats, candles, start, end) covering candles that start in [t0, t1)."""
    starts = [c[0] for c in candles]
    a = max(bisect.bisect_left(starts, t0), WARMUP)
    b = bisect.bisect_left(starts, t1)
    return feats, candles, a, b


def backtest_many(activate, windows, costs, chunks=1, record=False):
    """Same network on several coins, each with its own equal pot. Fitness is the mean."""
    per = {}
    for pid, (feats, candles, a, b) in windows.items():
        if b - a >= 24:
            per[pid] = backtest(activate, feats, candles, a, b, costs, chunks=chunks, record=record)
    if not per:
        raise RuntimeError("No coin has enough candles in this window")
    rs = list(per.values())
    n = len(rs)
    out = {
        "fitness": sum(r["fitness"] for r in rs) / n,
        "return_pct": sum(r["return_pct"] for r in rs) / n,
        "buy_hold_pct": sum(r["buy_hold_pct"] for r in rs) / n,
        "max_drawdown_pct": max(r["max_drawdown_pct"] for r in rs),
        "trades": sum(r["trades"] for r in rs),
        "fees_paid": sum(r["fees_paid"] for r in rs),
        "per_product": {p: {k: v for k, v in r.items() if k not in ("curve", "log")}
                        for p, r in per.items()},
    }
    if record:
        out["curve"] = combine_curves([r["curve"] for r in rs])
    return out


def combine_curves(curves):
    """Sum several (time, equity) curves, carrying each one's last value forward."""
    times = sorted({t for c in curves for t, _ in c})
    idx, last, out = [0] * len(curves), [c[0][1] if c else 0.0 for c in curves], []
    for t in times:
        for k, c in enumerate(curves):
            while idx[k] < len(c) and c[idx[k]][0] <= t:
                last[k] = c[idx[k]][1]
                idx[k] += 1
        out.append((t, sum(last)))
    return out
