"""Trades the current champion network on live Coinbase data, one decision per
closed candle. Paper mode by default; serves the dashboard on WEB_PORT."""
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
    quote_ccy = config.PRODUCT_ID.split("-")[1]
    accounts = coinbase_client().get_accounts(limit=250).to_dict()["accounts"]
    avail = sum(float(a["available_balance"]["value"]) for a in accounts if a["currency"] == quote_ccy)
    return min(avail, config.LIVE_MAX_USD)


def new_session(broker, taker_fee):
    start = live_budget() if broker.live else config.PAPER_START_USD
    if start < 1:
        raise SystemExit("Not enough quote currency in your Coinbase account to trade")
    _, asks = coinbase_data.get_order_book(config.PRODUCT_ID, limit=1)
    st = {"session": time.strftime("%Y%m%d-%H%M%S"), "mode": config.TRADING_MODE,
          "product": config.PRODUCT_ID, "started_at": time.time(), "start_equity": start,
          "cash": start, "qty": 0.0, "entry_cost": 0.0, "held": 0, "peak": start,
          "halted": False, "taker_fee": taker_fee,
          "bench_qty": buy_with_cash(start, asks[0][0] * (1 + config.SLIPPAGE_BPS / 10_000),
                                       taker_fee)[0]}
    log.info("Started new %s session %s with $%.2f", st["mode"], st["session"], start)
    return st


def make_broker(taker_fee):
    if config.TRADING_MODE != "live":
        return PaperBroker(taker_fee)
    if config.LIVE_CONFIRM != LIVE_CONFIRM_PHRASE:
        raise SystemExit(f"TRADING_MODE=live requires LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
    if not config.has_api_keys():
        raise SystemExit("TRADING_MODE=live requires COINBASE_API_KEY and COINBASE_API_SECRET")
    log.warning("LIVE TRADING ENABLED: real orders, budget capped at $%.2f", config.LIVE_MAX_USD)
    return CoinbaseBroker(taker_fee)


def wait_for_champion():
    while True:
        champ = load_champion()
        if champ:
            return champ
        log.info("No champion yet - waiting for the trainer's first cycle...")
        time.sleep(30)


def recent_candles():
    g = config.GRANULARITY
    now = int(time.time())
    end = now - now % g
    return coinbase_data.fetch_candles(config.PRODUCT_ID, g, end - (features.WARMUP + 150) * g, end)


def step(db, st, broker, net):
    candles = recent_candles()
    feats = features.market_features(candles, config.GRANULARITY)[-1]
    mark = broker.mark_price()
    fee = broker.taker_fee
    in_pos = st["qty"] > 0
    pnl = sell_base(st["qty"], mark, fee)[0] / st["entry_cost"] - 1 if in_pos else 0.0
    out = net.activate(list(feats) + features.position_features(in_pos, pnl, st["held"]))[0]
    action = decide(out, in_pos)

    stop = os.path.exists(config.KILL_SWITCH_PATH)
    if stop or st["halted"]:
        action = "sell" if in_pos else None

    fill = None
    if action == "buy":
        fill = broker.buy(st["cash"] * config.POSITION_FRACTION)
    elif action == "sell":
        fill = broker.sell(st["qty"])

    trade_pnl = None
    if fill:
        st["cash"] += fill["cash_delta"]
        if fill["side"] == "buy":
            st["qty"], st["entry_cost"], st["held"] = fill["qty"], -fill["cash_delta"], 0
        else:
            trade_pnl = fill["cash_delta"] - st["entry_cost"]
            st["qty"] = max(0.0, st["qty"] - fill["qty"])
            if st["qty"] * mark < 1:  # dust left by rounding
                st["qty"], st["entry_cost"] = 0.0, 0.0
    elif st["qty"] > 0:
        st["held"] += 1

    equity = st["cash"] + (sell_base(st["qty"], mark, fee)[0] if st["qty"] > 0 else 0.0)
    bench = sell_base(st["bench_qty"], mark, fee)[0]
    st["peak"] = max(st["peak"], equity)
    if broker.live and equity < st["peak"] * (1 - config.HALT_DRAWDOWN_PCT / 100) and not st["halted"]:
        st["halted"] = True
        log.error("Equity fell %.1f%% from peak - HALTING (will flatten next bar)", config.HALT_DRAWDOWN_PCT)
    st["last_output"], st["last_update"], st["last_price"] = out, time.time(), mark

    now = time.time()
    if fill:
        db.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            now, st["session"], st["mode"], fill["side"], fill["qty"], fill["price"], fill["fee"],
            fill["cash_delta"], trade_pnl, equity, out, fill.get("order_id")))
        log.info("%s %.8f @ %.2f  fee $%.4f  %s equity $%.2f", fill["side"].upper(), fill["qty"],
                 fill["price"], fill["fee"],
                 f"pnl ${trade_pnl:+.2f} " if trade_pnl is not None else "", equity)
    db.execute("INSERT INTO equity VALUES (?,?,?,?,?,?,?)",
               (now, st["session"], equity, bench, mark, out, int(st["qty"] > 0)))
    save_state(db, st)
    log.info("bar close  price %.2f  output %+.3f  equity $%.2f  (buy&hold $%.2f)%s",
             mark, out, equity, bench, "  [STOP file present]" if stop else "")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    threading.Thread(target=web.serve, daemon=True).start()

    db = open_db()
    taker_fee = fees.resolve_taker_fee()
    broker = make_broker(taker_fee)
    st = load_state(db)
    if not st or st["mode"] != config.TRADING_MODE or st["product"] != config.PRODUCT_ID:
        st = new_session(broker, taker_fee)
        save_state(db, st)
        mark = broker.mark_price()
        db.execute("INSERT INTO equity VALUES (?,?,?,?,?,?,?)", (
            time.time(), st["session"], st["cash"],
            sell_base(st["bench_qty"], mark, taker_fee)[0], mark, None, 0))
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
                champ, champ_mtime = load_champion() or champ, mtime
                net = neat.nn.FeedForwardNetwork.create(champ["genome"], cfg)
                st["champion"] = champ["meta"]
                log.info("Switched to new champion from generation %s", champ["meta"]["generation"])
            step(db, st, broker, net)
        except Exception:
            log.exception("Trading step failed; will retry next bar")


if __name__ == "__main__":
    main()
