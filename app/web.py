"""Small read-only dashboard: paper/live performance vs buy & hold, trades, training."""
import json
import os
import sqlite3

from flask import Flask, jsonify, render_template

from . import config

app = Flask(__name__)


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
    if not st:
        return jsonify({"ready": False, "product": config.PRODUCT_ID, "mode": config.TRADING_MODE})
    s = st["session"]
    eq = _q(config.TRADER_DB, "SELECT equity, benchmark FROM equity WHERE session=? ORDER BY ts", (s,))
    trades = _q(config.TRADER_DB, "SELECT side, fee, pnl FROM trades WHERE session=?", (s,))
    sells = [t for t in trades if t["side"] == "sell"]
    peak, max_dd = st["start_equity"], 0.0
    for r in eq:
        peak = max(peak, r["equity"])
        max_dd = max(max_dd, 1 - r["equity"] / peak)
    last = eq[-1] if eq else {"equity": st["start_equity"], "benchmark": st["start_equity"]}
    start = st["start_equity"]
    return jsonify({
        "ready": True, "mode": st["mode"], "product": st["product"], "session": s,
        "started_at": st["started_at"], "last_update": st.get("last_update"),
        "granularity": config.GRANULARITY, "taker_fee": st["taker_fee"],
        "slippage_bps": config.SLIPPAGE_BPS,
        "start_equity": start, "equity": last["equity"], "benchmark": last["benchmark"],
        "return_pct": 100 * (last["equity"] / start - 1),
        "benchmark_pct": 100 * (last["benchmark"] / start - 1),
        "fees_paid": sum(t["fee"] for t in trades), "fills": len(trades),
        "round_trips": len(sells),
        "win_rate": (sum(1 for t in sells if t["pnl"] > 0) / len(sells)) if sells else None,
        "max_drawdown_pct": 100 * max_dd, "in_position": st["qty"] > 0,
        "cash": st["cash"], "qty": st["qty"], "last_price": st.get("last_price"),
        "last_output": st.get("last_output"), "halted": st["halted"],
        "stop_file": os.path.exists(config.KILL_SWITCH_PATH),
        "champion": st.get("champion"),
    })


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
    return jsonify(_q(config.TRADER_DB, "SELECT ts, side, qty, price, fee, cash_delta, pnl, equity, output "
                      "FROM trades WHERE session=? ORDER BY ts DESC LIMIT 200", (st["session"],)))


@app.route("/api/training")
def training():
    runs = _q(config.TRAINER_DB, "SELECT * FROM runs ORDER BY ts DESC LIMIT 50")
    for r in runs:
        for k in ("train", "validation", "champion_validation", "costs", "val_curve"):
            r[k] = json.loads(r[k]) if r.get(k) else None
    gens = _q(config.TRAINER_DB, "SELECT generation, best, mean, species FROM generations "
              "ORDER BY rowid DESC LIMIT 400")
    return jsonify({"runs": runs, "generations": gens[::-1]})


def serve():
    from waitress import serve as waitress_serve
    waitress_serve(app, host="0.0.0.0", port=config.WEB_PORT, threads=4, _quiet=True)
