import math
import random

from app import features
from app.broker import walk_book
from app.sim import Costs, backtest, backtest_many, buy_with_cash, combine_curves, limit_fills, sell_base, window

TAKER = Costs(taker_fee=0.012, maker_fee=0.006, slippage_bps=5, spread_bps=2, order_type="taker")
MAKER = Costs(taker_fee=0.012, maker_fee=0.006, slippage_bps=5, spread_bps=2, order_type="maker",
              fallback_bars=2)
N = 900


def test_round_trip_costs_two_fees_plus_spread_and_slippage():
    qty, _, buy_fee = buy_with_cash(1000, TAKER.buy_price(100.0), TAKER.taker_fee)
    cash, _, sell_fee = sell_base(qty, TAKER.sell_price(100.0), TAKER.taker_fee)
    assert math.isclose(buy_fee + sell_fee, 23.7, rel_tol=0.01)
    assert 975 < cash < 976  # ~2.4% fees + ~0.12% spread/slippage


def _candles(n=N, seed=1, start=0):
    rnd, p, out = random.Random(seed), 100.0, []
    for i in range(n):
        o, p = p, p * math.exp(rnd.gauss(0, 0.01))
        out.append((start + i * 3600, o, max(o, p) * 1.002, min(o, p) * 0.998, p, rnd.uniform(1, 10)))
    return out


def test_features_shape_and_memory_input():
    candles = _candles()
    feats = features.market_features(candles, 3600)
    assert feats.shape == (N, features.N_MARKET)
    assert abs(feats[features.WARMUP:]).max() <= 1.0 + 1e-9
    assert features.position_features(True, 0.0, 0, -0.7)[-1] == -0.7
    assert features.N_INPUTS == 20


def test_do_nothing_keeps_cash_and_always_long_matches_buy_hold():
    candles = _candles()
    feats = features.market_features(candles, 3600)
    r = backtest(lambda x: [0.0], feats, candles, features.WARMUP, N, TAKER)
    assert r["final_equity"] == 1000 and r["trades"] == 0 and r["fees_paid"] == 0
    always_long = backtest(lambda x: [1.0], feats, candles, features.WARMUP, N, TAKER)
    assert math.isclose(always_long["return_pct"], always_long["buy_hold_pct"], abs_tol=1e-9)


def test_churning_bleeds_fees():
    candles = _candles()
    feats = features.market_features(candles, 3600)
    flip = iter(range(10**6))
    r = backtest(lambda x: [1.0 if next(flip) % 2 else -1.0], feats, candles, features.WARMUP, N, TAKER)
    assert r["trades"] > 250 and r["final_equity"] < 20


def test_limit_order_needs_a_trade_through():
    bar = (0, 100, 101, 99, 100, 1)
    assert limit_fills("buy", 99.5, bar) and not limit_fills("buy", 99.0, bar)
    assert limit_fills("sell", 100.5, bar) and not limit_fills("sell", 101.0, bar)


def test_maker_round_trip_is_cheaper_and_waits_for_next_bar():
    flat = [(i * 3600, 100.0, 101.0, 99.0, 100.0, 5.0) for i in range(N)]
    feats = features.market_features(flat, 3600)
    signals = iter([1.0] + [0.0] * 9 + [-1.0] + [0.0] * 10**6)
    m = backtest(lambda x: [next(signals)], feats, flat, features.WARMUP, N, MAKER, record=True)
    assert [e["side"] for e in m["log"]] == ["buy", "sell"]
    assert m["log"][0]["time"] == flat[features.WARMUP + 1][0]  # filled on the bar after the decision
    signals = iter([1.0] + [0.0] * 9 + [-1.0] + [0.0] * 10**6)
    t = backtest(lambda x: [next(signals)], feats, flat, features.WARMUP, N, TAKER)
    assert m["fees_paid"] < 0.55 * t["fees_paid"] and m["final_equity"] > t["final_equity"] + 10


def test_unfilled_maker_sell_falls_back_to_market():
    # flat until the buy fills, then a steady fall: each bar's high only touches the
    # previous close, so a resting sell at the ask never trades through
    w = features.WARMUP
    candles, p = [(i * 3600, 100.0, 100.5, 98.0, 100.0, 5.0) for i in range(w + 2)], 100.0
    for i in range(w + 2, N):
        o, p = p, p * 0.995
        candles.append((i * 3600, o, o * 1.0001, p * 0.999, p, 5.0))
    feats = features.market_features(candles, 3600)
    sig = iter([1.0] + [-1.0] * 10**6)
    r = backtest(lambda x: [next(sig)], feats, candles, w, N, MAKER, record=True)
    buy, sell = r["log"][:2]
    assert buy["side"] == "buy" and math.isclose(buy["fee"], 1000 / 1.006 * 0.006)  # maker fee
    # limit sells placed at bars w+1 and w+2 both miss -> market sell at bar w+3, taker fee
    assert sell["side"] == "sell" and sell["time"] == candles[w + 1 + MAKER.fallback_bars][0]
    assert sell["price"] < candles[w + 1 + MAKER.fallback_bars][4]  # sold below close: spread + slippage


def test_backtest_many_averages_coins_and_combines_curves():
    a, b = _candles(seed=1), _candles(seed=2)
    wins = {"A": window(features.market_features(a, 3600), a, 0, 10**9),
            "B": window(features.market_features(b, 3600), b, 0, 10**9)}
    r = backtest_many(lambda x: [1.0], wins, TAKER, record=True)
    singles = [backtest(lambda x: [1.0], *wins[k], TAKER) for k in "AB"]
    assert math.isclose(r["return_pct"], sum(s["return_pct"] for s in singles) / 2)
    assert math.isclose(r["curve"][-1][1], sum(s["final_equity"] for s in singles))
    # a coin with no point yet counts at its first (starting) value
    assert combine_curves([[(1, 10), (3, 30)], [(2, 5)]]) == [(1, 15), (2, 15), (3, 35)]


def test_walk_book_vwap():
    asks = [(100.0, 1.0), (101.0, 1.0)]
    assert math.isclose(walk_book(asks, quote=150.0), 150.0 / (1 + 50 / 101))
    bids = [(99.0, 1.0), (98.0, 1.0)]
    assert math.isclose(walk_book(bids, base=1.5), (99 + 49) / 1.5)
