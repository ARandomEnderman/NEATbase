import math
import random

from app import features
from app.broker import walk_book
from app.sim import Costs, backtest, buy_with_cash, sell_base


def test_round_trip_costs_two_fees_plus_spread_and_slippage():
    costs = Costs(taker_fee=0.012, slippage_bps=5, spread_bps=2)
    qty, _, buy_fee = buy_with_cash(1000, costs.buy_price(100.0), costs.taker_fee)
    cash, _, sell_fee = sell_base(qty, costs.sell_price(100.0), costs.taker_fee)
    assert math.isclose(buy_fee + sell_fee, 23.7, rel_tol=0.01)
    assert 975 < cash < 976  # ~2.4% fees + ~0.12% spread/slippage


def _candles(n=600, seed=1):
    rnd, p, out = random.Random(seed), 100.0, []
    for i in range(n):
        o, p = p, p * math.exp(rnd.gauss(0, 0.01))
        out.append((i * 3600, o, max(o, p) * 1.002, min(o, p) * 0.998, p, rnd.uniform(1, 10)))
    return out


def test_do_nothing_keeps_cash_and_buy_hold_matches_fees():
    candles = _candles()
    feats = features.market_features(candles, 3600)
    assert feats.shape == (600, features.N_MARKET)
    r = backtest(lambda x: [0.0], feats, candles, features.WARMUP, 600, Costs())
    assert r["final_equity"] == 1000 and r["trades"] == 0 and r["fees_paid"] == 0

    always_long = backtest(lambda x: [1.0], feats, candles, features.WARMUP, 600, Costs())
    assert math.isclose(always_long["return_pct"], always_long["buy_hold_pct"], abs_tol=1e-9)


def test_churning_bleeds_fees():
    candles = _candles()
    feats = features.market_features(candles, 3600)
    flip = iter(range(10**6))
    r = backtest(lambda x: [1.0 if next(flip) % 2 else -1.0], feats, candles,
                 features.WARMUP, 600, Costs())
    assert r["trades"] > 150 and r["final_equity"] < 100


def test_walk_book_vwap():
    asks = [(100.0, 1.0), (101.0, 1.0)]
    assert math.isclose(walk_book(asks, quote=150.0), 150.0 / (1 + 50 / 101))
    bids = [(99.0, 1.0), (98.0, 1.0)]
    assert math.isclose(walk_book(bids, base=1.5), (99 + 49) / 1.5)
