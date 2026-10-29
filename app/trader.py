"""Trades the current champion network on live Coinbase data, one decision per closed
candle for every coin in PRODUCT_IDS. Each coin has its own equal share ("slot") of the
money. Paper mode by default; serves the dashboard on WEB_PORT."""
import json
import logging
import os
import sqlite3
import threading
import time

import neat

from . import coinbase_data, config, features, fees, web
from .broker import CoinbaseBroker, PaperBroker
from .sim import buy_with_cash, decide, sell_base
from .trainer import load_champion, load_neat_config

log = logging.getLogger("trader")
LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_THIS_USES_REAL_MONEY"


def open_db():
    db = sqlite3.connect(config.TRADER_DB, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS state (k TEXT PRIMARY KEY, v TEXT)")
    db.execute("""CREATE TABLE IF NOT EXISTS trades (
        ts REAL, session TEXT, mode TEXT, side TEXT, qty REAL, price REAL, fee REAL,
        cash_delta REAL, pnl REAL, equity REAL, output REAL, order_id TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS equity (
        ts REAL, session TEXT, equity REAL, benchmark REAL, price REAL, output REAL,
        in_position INTEGER)""")
    for col, typ in (("product", "TEXT"), ("order_type", "TEXT")):
        if col not in [r[1] for r in db.execute("PRAGMA table_info(trades)")]:
            db.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")
    db.commit()
    return db


def load_state(db):
    row = db.execute("SELECT v FROM state WHERE k='portfolio'").fetchone()
    return json.loads(row[0]) if row else None


def save_state(db, st):
    db.execute("INSERT OR REPLACE INTO state VALUES ('portfolio', ?)", (json.dumps(st),))
    db.commit()


def live_budget():
    from .fees import coinbase_client
    quote_ccy = config.PRODUCT_IDS[0].split("-")[1]
    accounts = coinbase_client().get_accounts(limit=250).to_dict()["accounts"]
    avail = sum(float(a["available_balance"]["value"]) for a in accounts if a["currency"] == quote_ccy)
    return min(avail, config.LIVE_MAX_USD)


def new_session(broker):
    start = live_budget() if broker.live else config.PAPER_START_USD
    n = len(config.PRODUCT_IDS)
    if start / n < 2:
        raise SystemExit(f"Not enough quote currency to split across {n} coins")
    slots = {}
    for pid in config.PRODUCT_IDS:
        _, asks = coinbase_data.get_order_book(pid, limit=1)
        slots[pid] = {"cash": start / n, "qty": 0.0, "entry_cost": 0.0, "held": 0, "prev_out": 0.0,
                      "pending": None, "sell_fails": 0, "last_output": None, "last_price": None,
                      "bench_qty": buy_with_cash(start / n, asks[0][0] * (1 + config.SLIPPAGE_BPS / 10_000),
                                                 broker.taker_fee)[0]}
    st = {"session": time.strftime("%Y%m%d-%H%M%S"), "mode": config.TRADING_MODE,
          "products": config.PRODUCT_IDS, "started_at": time.time(), "start_equity": start,
          "peak": start, "halted": False, "taker_fee": broker.taker_fee, "maker_fee": broker.maker_fee,
          "order_type": config.ORDER_TYPE, "slots": slots}
    log.info("Started new %s session %s with $%.2f across %s", st["mode"], st["session"], start,
             ", ".join(config.PRODUCT_IDS))
    return st


def make_broker(taker_fee, maker_fee):
    if config.TRADING_MODE != "live":
        return PaperBroker(taker_fee, maker_fee)
    if config.LIVE_CONFIRM != LIVE_CONFIRM_PHRASE:
        raise SystemExit(f"TRADING_MODE=live requires LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
    if not config.has_api_keys():
        raise SystemExit("TRADING_MODE=live requires an API key (config/cdp_api_key.json)")
    log.warning("LIVE TRADING ENABLED: real orders, budget capped at $%.2f", config.LIVE_MAX_USD)
    return CoinbaseBroker(taker_fee, maker_fee)


def wait_for_champion():
    while True:
        champ = load_champion()
        if champ:
            return champ
        log.info("No champion yet - waiting for the trainer's first cycle...")
        time.sleep(30)


def recent_candles(pid):
    g = config.GRANULARITY
    now = int(time.time())
    end = now - now % g
    return coinbase_data.fetch_candles(pid, g, end - (features.WARMUP + 300) * g, end)


def apply_fill(slot, fill, mark):
    """Book a (possibly partial) fill into a slot. Returns realised P&L for sells."""
    slot["cash"] += fill["cash_delta"]
    if fill["side"] == "buy":
        if slot["qty"] <= 0:
            slot["held"] = 0
        slot["qty"] += fill["qty"]
        slot["entry_cost"] += -fill["cash_delta"]
        return None
    frac = min(1.0, fill["qty"] / slot["qty"]) if slot["qty"] > 0 else 1.0
    cost = slot["entry_cost"] * frac
    slot["qty"] = max(0.0, slot["qty"] - fill["qty"])
    slot["entry_cost"] -= cost
    if slot["qty"] * mark < 1:  # dust left by rounding
        slot["qty"], slot["entry_cost"] = 0.0, 0.0
    return fill["cash_delta"] - cost


def step_product(db, st, broker, net, pid, forced_exit):
    slot = st["slots"][pid]
    candles = recent_candles(pid)
    mark = broker.mark_price(pid)
    fills = []

    if slot["pending"]:  # maker order placed at the previous close
        order, slot["pending"] = slot["pending"], None
        fill = broker.resolve_limit(pid, order, candles)
        if fill:
            fills.append((fill, "maker"))
        if order["side"] == "sell" and (not fill or slot["qty"] - fill["qty"] > 0):
            slot["sell_fails"] += 1
        for f, _ in fills:
            f["pnl"] = apply_fill(slot, f, mark)

    feats = features.market_features(candles, config.GRANULARITY)[-1]
    fee = broker.taker_fee
    in_pos = slot["qty"] > 0
    pnl = sell_base(slot["qty"], mark, fee)[0] / slot["entry_cost"] - 1 if in_pos and slot["entry_cost"] else 0.0
    out = net.activate(list(feats) + features.position_features(in_pos, pnl, slot["held"], slot["prev_out"]))[0]
    slot["prev_out"] = out
    action = decide(out, in_pos)
    if forced_exit:
        action = "sell" if in_pos else None
    if action != "sell":
        slot["sell_fails"] = 0

    maker = config.ORDER_TYPE == "maker" and not forced_exit
    bar = candles[-1][0]
    if action == "buy":
        spend = slot["cash"] * config.POSITION_FRACTION
        if maker:
            slot["pending"] = broker.place_limit(pid, "buy", spend, bar)
        else:
            f = broker.buy(pid, spend)
            if f:
                f["pnl"] = apply_fill(slot, f, mark)
                fills.append((f, "taker"))
    elif action == "sell":
        if maker and slot["sell_fails"] < config.MAKER_SELL_FALLBACK_BARS:
            slot["pending"] = broker.place_limit(pid, "sell", slot["qty"], bar)
        else:
            f = broker.sell(pid, slot["qty"])
            if f:
                f["pnl"] = apply_fill(slot, f, mark)
                fills.append((f, "taker"))
                slot["sell_fails"] = 0
    elif slot["qty"] > 0:
        slot["held"] += 1

    slot["last_output"], slot["last_price"] = out, mark
    slot_equity = slot["cash"] + (sell_base(slot["qty"], mark, fee)[0] if slot["qty"] > 0 else 0.0)
    for f, kind in fills:
        db.execute("INSERT INTO trades (ts, session, mode, side, qty, price, fee, cash_delta, pnl, equity, "
                   "output, order_id, product, order_type) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                       time.time(), st["session"], st["mode"], f["side"], f["qty"], f["price"], f["fee"],
                       f["cash_delta"], f["pnl"], slot_equity, out, f.get("order_id"), pid, kind))
        log.info("%s %s %.8f @ %.4f (%s) fee $%.4f %s", pid, f["side"].upper(), f["qty"], f["price"], kind,
                 f["fee"], f"pnl ${f['pnl']:+.2f}" if f["pnl"] is not None else "")
    pend = slot["pending"]
    log.info("%s close %.4f output %+.3f %s%s", pid, mark, out, "LONG" if slot["qty"] > 0 else "cash",
             f" | resting {pend['side']} @ {pend['price']:.4f}" if pend else "")
    return slot_equity, sell_base(slot["bench_qty"], mark, fee)[0]


def step(db, st, broker, net):
    stop = os.path.exists(config.KILL_SWITCH_PATH)
    forced = stop or st["halted"]
    equity = bench = 0.0
    for pid in st["products"]:
        try:
            e, b = step_product(db, st, broker, net, pid, forced)
        except Exception:
            log.exception("%s: step failed; keeping its last known value", pid)
            slot = st["slots"][pid]
            px = slot["last_price"] or 0.0
            e = slot["cash"] + (sell_base(slot["qty"], px, broker.taker_fee)[0] if slot["qty"] > 0 else 0.0)
            b = sell_base(slot["bench_qty"], px, broker.taker_fee)[0]
        equity, bench = equity + e, bench + b

    st["peak"] = max(st["peak"], equity)
    if broker.live and not st["halted"] and equity < st["peak"] * (1 - config.HALT_DRAWDOWN_PCT / 100):
        st["halted"] = True
        log.error("Equity fell %.1f%% from peak - HALTING (flattening next bar)", config.HALT_DRAWDOWN_PCT)
    st["last_update"] = time.time()
    n_long = sum(1 for s in st["slots"].values() if s["qty"] > 0)
    db.execute("INSERT INTO equity VALUES (?,?,?,?,?,?,?)",
               (time.time(), st["session"], equity, bench, None, None, n_long))
    save_state(db, st)
    log.info("bar close: equity $%.2f (buy&hold $%.2f), %d/%d coins held%s", equity, bench, n_long,
             len(st["products"]), "  [PAUSED]" if stop else "")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    threading.Thread(target=web.serve, daemon=True).start()

    db = open_db()
    broker = make_broker(*fees.resolve_fees())
    st = load_state(db)
    if not st or st.get("mode") != config.TRADING_MODE or st.get("products") != config.PRODUCT_IDS:
        st = new_session(broker)
        save_state(db, st)
        db.execute("INSERT INTO equity VALUES (?,?,?,?,?,?,?)", (
            time.time(), st["session"], st["start_equity"],
            sum(sell_base(s["bench_qty"], broker.mark_price(p), broker.taker_fee)[0]
                for p, s in st["slots"].items()), None, None, 0))
        db.commit()

    cfg = load_neat_config()
    champ = wait_for_champion()
    champ_mtime = os.path.getmtime(config.CHAMPION_PATH)
    net = neat.nn.FeedForwardNetwork.create(champ["genome"], cfg)
    st["champion"] = champ["meta"]
    save_state(db, st)

    g = config.GRANULARITY
    while True:
        next_close = (int(time.time()) // g + 1) * g
        time.sleep(max(0, next_close + 5 - time.time()))  # give Coinbase a moment to publish the candle
        try:
            mtime = os.path.getmtime(config.CHAMPION_PATH)
            if mtime != champ_mtime:
                new = load_champion()
                if new:
                    champ, net = new, neat.nn.FeedForwardNetwork.create(new["genome"], cfg)
                    st["champion"] = champ["meta"]
                    log.info("Switched to new champion (%s)", champ["meta"].get("how"))
                champ_mtime = mtime
            step(db, st, broker, net)
        except Exception:
            log.exception("Trading step failed; will retry next bar")


if __name__ == "__main__":
    main()
