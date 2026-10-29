import math
import random

import numpy as np

from app import flowcheck
from app.recorder import Recorder, book_features, open_db


def trade(tid, side, price, size, t="2026-10-29T17:30:10.123456789Z", pid="BTC-USD"):
    return {"product_id": pid, "trade_id": str(tid), "side": side, "price": str(price), "size": str(size), "time": t}


def test_book_features():
    bids = {99.99: 2.0, 99.5: 3.0, 98.0: 10.0}
    asks = {100.01: 1.0, 100.4: 1.0, 102.0: 10.0}
    f = book_features(bids, asks)
    assert math.isclose(f["spread_bps"], 0.02 / 100 * 1e4)
    assert math.isclose(f["imb_10bps"], (2 - 1) / 3)  # within 0.1% of mid: 99.99 vs 100.01
    assert math.isclose(f["imb_50bps"], (5 - 2) / 7)
    assert f["microprice_bps"] > 0  # less size on the ask -> price leans up
    assert book_features({}, asks) is None and book_features({101.0: 1}, {100.0: 1}) is None


def test_aggressor_flip_dedupe_missed_and_snapshot(tmp_path):
    r = Recorder(["BTC-USD"], open_db(str(tmp_path / "flow.db")))
    r.on_trades([trade(10, "BUY", 100, 1), trade(9, "SELL", 100, 1)], snapshot=True)  # only learns ids
    assert r.minutes == {} and r.last_trade_id["BTC-USD"] == 10
    # newest first, like Coinbase; 11 is missing
    r.on_trades([trade(13, "BUY", 100, 2), trade(12, "SELL", 101, 1), trade(10, "SELL", 100, 5)])
    (key, m), = r.minutes.items()
    assert key == ("BTC-USD", 1793295000)
    assert m.buy_vol == 1 and m.buy_notional == 101  # "SELL" print = aggressive buyer
    assert m.sell_vol == 2 and m.n_sell == 1           # "BUY" print = aggressive seller
    assert m.missed == 1                               # id 11 never arrived; 10 was a duplicate


def test_late_trades_never_overwrite_a_written_minute(tmp_path):
    db = open_db(str(tmp_path / "flow.db"))
    r = Recorder(["BTC-USD"], db)
    r.last_trade_id["BTC-USD"] = 0
    r.on_trades([trade(1, "SELL", 100, 1, t="2026-10-29T17:30:10Z")])
    r.flush(1793295000 + 120)
    r.on_trades([trade(2, "SELL", 100, 3, t="2026-10-29T17:30:50Z")])  # late, for the written minute
    r.flush(1793295000 + 200)
    rows = db.execute("SELECT minute, buy_vol FROM flow ORDER BY minute").fetchall()
    assert rows == [(1793295000, 1.0), (1793295060, 3.0)]


def test_analyze_finds_a_planted_signal_and_ignores_noise():
    rnd = random.Random(0)
    samples = []
    for i in range(3000):
        sig, noise = rnd.gauss(0, 1), rnd.gauss(0, 1)
        y1 = 0.002 * sig + rnd.gauss(0, 0.01)
        samples.append((i * 3600, {"signal": sig, "noise": noise}, y1, y1 * 2 + rnd.gauss(0, 0.01)))
    res = {r["name"]: r for r in flowcheck.analyze(samples, fee_bps=120)}
    assert res["signal"]["consistent"] and res["signal"]["ic"] > 0.1
    assert not res["noise"]["consistent"] and abs(res["noise"]["ic"]) < 0.05


def test_flow_bars_need_enough_recorded_minutes():
    rows = [(m * 60, 10.0, 5.0, 1, 1, 0.1, 0.0, 0.0, 0.0, 1.0, 0) for m in range(55)]      # bar 0: 55 min
    rows += [(3600 + m * 60, 1.0, 1.0, 1, 1, 0.1, 0.0, 0.0, 0.0, 1.0, 0) for m in range(30)]  # bar 1: 30 min
    bars = flowcheck.bars_from_flow(rows, 3600)
    assert list(bars) == [0]
    f = flowcheck.features(bars, 3600)[0]
    assert math.isclose(f["flow_imbalance_1"], (550 - 275) / 825)
    assert np.isfinite(list(f.values())).all()
