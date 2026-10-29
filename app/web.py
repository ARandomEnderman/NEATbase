"""Dashboard: paper/live performance vs buy & hold, trades, training, and a pause switch."""
import json
import os
import sqlite3
import threading
import time

from flask import Flask, abort, jsonify, render_template, request

from . import auth, coinbase_data, config

app = Flask(__name__)
auth.install(app)


def _mask(key):
    return f"…{key[-6:]}" if key else None


def _q(path, sql, args=()):
    if not os.path.exists(path):
        return []
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, args).fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()


def _state():
    rows = _q(config.TRADER_DB, "SELECT v FROM state WHERE k='portfolio'")
    return json.loads(rows[0]["v"]) if rows else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def summary():
    st = _state()
    keys = {"api_key": _mask(config.COINBASE_API_KEY), "api_key_source": config.API_KEY_SOURCE,
            "stop_file": os.path.exists(config.KILL_SWITCH_PATH)}
    if not st:
        return jsonify({"ready": False, "products": config.PRODUCT_IDS, "mode": config.TRADING_MODE,
                        "recorder": _recorder_status(), **keys})
    s = st["session"]
    eq = _q(config.TRADER_DB, "SELECT equity, benchmark FROM equity WHERE session=? ORDER BY ts", (s,))
    trades = _q(config.TRADER_DB, "SELECT side, fee, pnl, order_type FROM trades WHERE session=?", (s,))
    sells = [t for t in trades if t["side"] == "sell"]
    peak, max_dd = st["start_equity"], 0.0
    for r in eq:
        peak = max(peak, r["equity"])
        max_dd = max(max_dd, 1 - r["equity"] / peak)
    last = eq[-1] if eq else {"equity": st["start_equity"], "benchmark": st["start_equity"]}
    start = st["start_equity"]
    return jsonify({
        "ready": True, "mode": st["mode"], "products": st["products"], "session": s,
        "started_at": st["started_at"], "last_update": st.get("last_update"),
        "granularity": config.GRANULARITY, "taker_fee": st["taker_fee"], "maker_fee": st["maker_fee"],
        "order_type": st["order_type"], "slippage_bps": config.SLIPPAGE_BPS,
        "max_position_fraction": config.MAX_POSITION_FRACTION,
        "start_equity": start, "equity": last["equity"], "benchmark": last["benchmark"],
        "return_pct": 100 * (last["equity"] / start - 1),
        "benchmark_pct": 100 * (last["benchmark"] / start - 1),
        "fees_paid": sum(t["fee"] for t in trades), "fills": len(trades),
        "maker_fills": sum(1 for t in trades if t["order_type"] == "maker"),
        "round_trips": len(sells),
        "win_rate": (sum(1 for t in sells if t["pnl"] > 0) / len(sells)) if sells else None,
        "max_drawdown_pct": 100 * max_dd, "halted": st["halted"],
        "slots": [{"product": p, "qty": sl["qty"], "cash": sl["cash"], "last_price": sl["last_price"],
                   "last_output": sl["last_output"],
                   "pending": f"{sl['pending']['side']} @ {sl['pending']['price']:g}" if sl["pending"] else None}
                  for p, sl in st["slots"].items()],
        "champion": st.get("champion"), "recorder": _recorder_status(), **keys,
    })


def _recorder_status():
    path = os.path.join(config.DATA_DIR, "flow.db")
    rows = _q(path, "SELECT COUNT(*) AS minutes, MIN(minute) AS first, MAX(minute) AS last, "
              "SUM(n_buy + n_sell) AS trades, SUM(missed_trades) AS missed FROM flow")
    if not rows or not rows[0]["minutes"]:
        return None
    r = rows[0]
    n_products = max(1, len(config.PRODUCT_IDS))
    span = max(60, r["last"] - r["first"] + 60)
    return {"minutes": r["minutes"], "first": r["first"], "last": r["last"], "days": span / 86400,
            "coverage": min(1.0, r["minutes"] / n_products / (span / 60)),
            "missed_pct": 100 * (r["missed"] or 0) / max(1, (r["trades"] or 0) + (r["missed"] or 0))}


@app.route("/api/pause", methods=["POST"])
def pause():
    """Same as the STOP file: flatten at the next candle and stop buying."""
    if (request.get_json(silent=True, force=True) or {}).get("paused"):
        open(config.KILL_SWITCH_PATH, "a").close()
    elif os.path.exists(config.KILL_SWITCH_PATH):
        os.remove(config.KILL_SWITCH_PATH)
    return jsonify({"paused": os.path.exists(config.KILL_SWITCH_PATH)})


@app.route("/api/equity")
def equity():
    st = _state()
    if not st:
        return jsonify([])
    rows = _q(config.TRADER_DB, "SELECT ts, equity, benchmark, price FROM equity WHERE session=? ORDER BY ts",
              (st["session"],))
    step = max(1, len(rows) // 1500)
    return jsonify(rows[::step] + (rows[-1:] if rows and (len(rows) - 1) % step else []))


@app.route("/api/trades")
def trades():
    st = _state()
    if not st:
        return jsonify([])
    return jsonify(_q(config.TRADER_DB, "SELECT ts, product, side, order_type, qty, price, fee, cash_delta, "
                      "pnl, equity, output FROM trades WHERE session=? ORDER BY ts DESC LIMIT 200",
                      (st["session"],)))


_prices, _prices_lock = {}, threading.Lock()


def _price_history(pid, t0):
    """Closed candles from Coinbase, cached until the next candle closes."""
    g = config.GRANULARITY
    now = int(time.time())
    bar = now - now % g
    with _prices_lock:
        hit = _prices.get(pid)
        if hit and hit["t0"] <= t0 and hit["bar"] == bar:
            return hit["candles"]
    candles = coinbase_data.fetch_candles(pid, g, t0, bar)
    with _prices_lock:
        _prices[pid] = {"t0": t0, "bar": bar, "candles": candles}
    return candles


@app.route("/api/prices/<pid>")
def prices(pid):
    """A coin's price since a few days before the session started, plus the bot's fills."""
    if pid not in config.PRODUCT_IDS:
        abort(404)
    st = _state()
    started = st["started_at"] if st else time.time()
    g = config.GRANULARITY
    t0 = int(max(started - 3 * 86400, time.time() - 60 * 86400))
    t0 -= t0 % g
    try:
        candles = _price_history(pid, t0)
    except Exception:
        candles = []
    fills = _q(config.TRADER_DB, "SELECT ts, side, order_type, qty, price, fee, cash_delta, pnl FROM trades "
               "WHERE session=? AND product=? ORDER BY ts", (st["session"], pid)) if st else []
    return jsonify({"product": pid, "granularity": g,
                    "candles": [[c[0] + g, c[4]] for c in candles],  # plotted at the candle's close
                    "trades": fills})


@app.route("/api/training")
def training():
    runs = _q(config.TRAINER_DB, "SELECT * FROM runs ORDER BY ts DESC LIMIT 50")
    for r in runs:
        for k in ("train", "validation", "champion_validation", "costs", "val_curve"):
            r[k] = json.loads(r[k]) if r.get(k) else None
    gens = _q(config.TRAINER_DB, "SELECT generation, best, mean, species FROM generations "
              "ORDER BY rowid DESC LIMIT 400")
    challengers = _q(config.TRAINER_DB, "SELECT id, born, generation, nodes, conns, validation, status, "
                     "forward, champion_forward, decided FROM challengers ORDER BY born DESC LIMIT 30")
    for c in challengers:
        for k in ("validation", "forward", "champion_forward"):
            c[k] = json.loads(c[k]) if c.get(k) else None
            if c[k]:
                c[k].pop("per_product", None)
        if c["status"] == "waiting":
            c["eligible_at"] = c["born"] + config.PROBATION_DAYS * 86400
    return jsonify({"runs": runs, "generations": gens[::-1], "challengers": challengers,
                    "probation_days": config.PROBATION_DAYS})


def serve():
    from waitress import serve as waitress_serve
    waitress_serve(app, host="0.0.0.0", port=config.WEB_PORT, threads=4, _quiet=True)
