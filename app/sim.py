"""Fee-aware trading simulator shared by training (backtests) and paper trading.

Every fill pays the taker fee (market orders) and loses the spread plus slippage.
Equity is always marked at *liquidation value*: what you would actually get if you
market-sold right now, after the spread, slippage and the sell fee.
"""
import math
from dataclasses import dataclass

from . import config
from .features import position_features


@dataclass
class Costs:
    taker_fee: float = config.TAKER_FEE
    slippage_bps: float = config.SLIPPAGE_BPS
    spread_bps: float = config.SPREAD_BPS

    def buy_price(self, mid):
        return mid * (1 + self.spread_bps / 20_000 + self.slippage_bps / 10_000)

    def sell_price(self, mid):
        return mid * (1 - self.spread_bps / 20_000 - self.slippage_bps / 10_000)


def buy_with_cash(cash, price, fee_rate):
    """Spend `cash` in total (fee included). Returns (base_qty, notional, fee)."""
    notional = cash / (1 + fee_rate)
    return notional / price, notional, notional * fee_rate


def sell_base(qty, price, fee_rate):
    """Sell `qty`. Returns (cash_received, notional, fee)."""
    notional = qty * price
    fee = notional * fee_rate
    return notional - fee, notional, fee


def decide(output, in_position):
    if not in_position and output > config.BUY_THRESHOLD:
        return "buy"
    if in_position and output < config.SELL_THRESHOLD:
        return "sell"
    return None


def backtest(activate, feats, candles, start, end, costs, start_cash=1000.0,
             fraction=config.POSITION_FRACTION, chunks=1, record=False):
    """Run a policy over bars [start, end). Decisions use bar i's closed candle and
    fill at that close (with costs)."""
    cash, qty, entry_cost, held = start_cash, 0.0, 0.0, 0
    trades, wins, fees = 0, 0, 0.0
    peak, max_dd = start_cash, 0.0
    bounds = [start + (end - start) * k // chunks for k in range(chunks + 1)]
    chunk_equity = [start_cash]
    curve, log = [], []

    for i in range(start, end):
        close = candles[i][4]
        sell_px = costs.sell_price(close)
        in_pos = qty > 0
        pnl = 0.0
        if in_pos:
            pnl = sell_base(qty, sell_px, costs.taker_fee)[0] / entry_cost - 1
        out = activate(list(feats[i]) + position_features(in_pos, pnl, held))[0]
        action = decide(out, in_pos)

        if action == "buy" and cash * fraction >= 1.0:
            spend = cash * fraction
            q, _, fee = buy_with_cash(spend, costs.buy_price(close), costs.taker_fee)
            cash -= spend
            qty, entry_cost, held, fees = q, spend, 0, fees + fee
            if record:
                log.append({"time": candles[i][0], "side": "buy", "price": close, "fee": fee})
        elif action == "sell":
            got, _, fee = sell_base(qty, sell_px, costs.taker_fee)
            cash += got
            trades += 1
            wins += got > entry_cost
            fees += fee
            if record:
                log.append({"time": candles[i][0], "side": "sell", "price": close, "fee": fee,
                            "pnl": got - entry_cost})
            qty, held = 0.0, 0
        elif qty > 0:
            held += 1

        equity = cash + (sell_base(qty, sell_px, costs.taker_fee)[0] if qty > 0 else 0.0)
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
        if i + 1 in bounds[1:]:
            chunk_equity.append(equity)
        if record:
            curve.append((candles[i][0], equity))

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
