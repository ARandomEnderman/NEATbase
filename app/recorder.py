"""Records order flow and order-book state from Coinbase's public WebSocket, one row per
coin per minute, into DATA_DIR/flow.db. Coinbase offers no history for this data, so
it has to be collected from now on; app.flowcheck later tests whether it predicts
anything.

Per minute and coin:
- aggressor flow: base volume, notional and trade count bought and sold by takers,
  the largest trade, VWAP, and trades missed (gaps in Coinbase's trade ids)
- order book, sampled every BOOK_SAMPLE_SECONDS and averaged over the minute: spread,
  bid/ask imbalance within 10, 50 and 100 bps of mid, USD depth within 100 bps, and
  the microprice offset from mid

Coinbase's trade `side` is the resting (maker) order's side, so the aggressor is the
opposite: a "SELL" print is a buyer lifting the ask (checked against the live book).
"""
import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime

import websockets

from . import config

log = logging.getLogger("recorder")
WS_URL = "wss://advanced-trade-ws.coinbase.com"
BOOK_SAMPLE_SECONDS = 5
FLUSH_GRACE_SECONDS = 5  # wait this long after a minute ends for late trades
DEPTHS_BPS = (10, 50, 100)
FLOW_DB = os.path.join(config.DATA_DIR, "flow.db")
COLUMNS = ["buy_vol", "sell_vol", "buy_notional", "sell_notional", "n_buy", "n_sell", "max_trade_notional",
           "vwap", "missed_trades", "spread_bps", "imb_10bps", "imb_50bps", "imb_100bps", "depth_100bps_usd",
           "microprice_bps", "book_samples"]


def open_db(path=FLOW_DB):
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(f"""CREATE TABLE IF NOT EXISTS flow (product TEXT, minute INTEGER,
        {", ".join(c + " REAL" for c in COLUMNS)}, PRIMARY KEY (product, minute))""")
    db.execute("CREATE TABLE IF NOT EXISTS status (k TEXT PRIMARY KEY, v TEXT)")
    db.commit()
    return db


def _ts(s):
    """Coinbase RFC3339 with nanoseconds -> epoch seconds."""
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".")
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s + "+00:00").timestamp()


def book_features(bids, asks):
    """bids/asks: {price: size}. Returns the per-sample book features, or None."""
    if not bids or not asks:
        return None
    bb, ba = max(bids), min(asks)
    if ba <= bb:
        return None
    mid = (bb + ba) / 2
    out = {"spread_bps": (ba - bb) / mid * 1e4}
    depth = {}
    for d in DEPTHS_BPS:
        lo, hi = mid * (1 - d / 1e4), mid * (1 + d / 1e4)
        b = sum(q for p, q in bids.items() if p >= lo)
        a = sum(q for p, q in asks.items() if p <= hi)
        out[f"imb_{d}bps"] = (b - a) / (b + a) if b + a > 0 else 0.0
        depth[d] = (b + a) * mid
    out["depth_100bps_usd"] = depth[100]
    qb, qa = bids[bb], asks[ba]
    micro = (bb * qa + ba * qb) / (qa + qb)  # leans toward the side with less size
    out["microprice_bps"] = (micro - mid) / mid * 1e4
    return out


class Minute:
    def __init__(self):
        self.buy_vol = self.sell_vol = self.buy_notional = self.sell_notional = 0.0
        self.n_buy = self.n_sell = 0
        self.max_trade_notional = 0.0
        self.missed = 0
        self.book = []

    def add_trade(self, aggressor_buy, price, size):
        n = price * size
        if aggressor_buy:
            self.buy_vol += size
            self.buy_notional += n
            self.n_buy += 1
        else:
            self.sell_vol += size
            self.sell_notional += n
            self.n_sell += 1
        self.max_trade_notional = max(self.max_trade_notional, n)

    def row(self):
        vol = self.buy_vol + self.sell_vol
        r = {"buy_vol": self.buy_vol, "sell_vol": self.sell_vol, "buy_notional": self.buy_notional,
             "sell_notional": self.sell_notional, "n_buy": self.n_buy, "n_sell": self.n_sell,
             "max_trade_notional": self.max_trade_notional,
             "vwap": (self.buy_notional + self.sell_notional) / vol if vol > 0 else None,
             "missed_trades": self.missed, "book_samples": len(self.book)}
        for k in ("spread_bps", "imb_10bps", "imb_50bps", "imb_100bps", "depth_100bps_usd", "microprice_bps"):
            vals = [s[k] for s in self.book]
            r[k] = sum(vals) / len(vals) if vals else None
        return r


class Recorder:
    def __init__(self, products, db):
        self.products = products
        self.db = db
        self.books = {}
        self.minutes = {}  # (product, minute start) -> Minute
        self.last_trade_id = {}
        self.flushed_upto = {}  # product -> first minute not yet written
        self.reconnects = 0

    def minute(self, pid, t):
        start = int(t // 60 * 60)
        start = max(start, self.flushed_upto.get(pid, start))  # late arrivals go to the open minute
        key = (pid, start)
        if key not in self.minutes:
            self.minutes[key] = Minute()
        return self.minutes[key]

    def on_trades(self, trades, snapshot=False):
        if snapshot:
            # first time we see a coin: start after its newest trade, so trades from before
            # the recorder was running aren't counted (on reconnects they are, with gaps)
            newest = {}
            for t in trades:
                newest[t["product_id"]] = max(newest.get(t["product_id"], 0), int(t["trade_id"]))
            for pid, tid in newest.items():
                self.last_trade_id.setdefault(pid, tid)
        # Coinbase lists trades newest first; replay them in id order
        for t in sorted(trades, key=lambda t: int(t["trade_id"])):
            pid, tid = t["product_id"], int(t["trade_id"])
            last = self.last_trade_id.get(pid)
            if last is not None and tid <= last:
                continue  # already counted (the snapshot on connect repeats recent trades)
            m = self.minute(pid, _ts(t["time"]))
            if last is not None and tid > last + 1:
                m.missed += tid - last - 1
            self.last_trade_id[pid] = tid
            m.add_trade(t["side"] == "SELL", float(t["price"]), float(t["size"]))

    def on_book(self, event):
        pid = event["product_id"]
        if event["type"] == "snapshot":
            self.books[pid] = ({}, {})
        bids, asks = self.books.setdefault(pid, ({}, {}))
        for u in event["updates"]:
            side = bids if u["side"] == "bid" else asks
            p, q = float(u["price_level"]), float(u["new_quantity"])
            if q == 0:
                side.pop(p, None)
            else:
                side[p] = q

    def sample_books(self, now):
        for pid, (bids, asks) in self.books.items():
            f = book_features(bids, asks)
            if f:
                self.minute(pid, now).book.append(f)

    def flush(self, now):
        done = [k for k in self.minutes if k[1] + 60 + FLUSH_GRACE_SECONDS <= now]
        for pid, start in sorted(done, key=lambda k: k[1]):
            r = self.minutes.pop((pid, start)).row()
            self.flushed_upto[pid] = max(self.flushed_upto.get(pid, 0), start + 60)
            self.db.execute(f"INSERT OR REPLACE INTO flow (product, minute, {', '.join(COLUMNS)}) VALUES "
                            f"(?, ?, {', '.join('?' * len(COLUMNS))})", [pid, start] + [r[c] for c in COLUMNS])
        if done:
            self.db.execute("INSERT OR REPLACE INTO status VALUES ('last_flush', ?)", (str(now),))
            self.db.commit()

    async def run_once(self):
        async with websockets.connect(WS_URL, max_size=2 ** 25, ping_interval=20, ping_timeout=20,
                                      open_timeout=20) as ws:
            for ch in ("heartbeats", "market_trades", "level2"):
                await ws.send(json.dumps({"type": "subscribe", "product_ids": self.products, "channel": ch}))
            log.info("Connected; recording %s", ", ".join(self.products))
            self.books = {}
            seq = None
            next_sample = time.time()
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                s = msg.get("sequence_num")
                if s is not None:
                    if seq is not None and s != seq + 1:
                        raise ConnectionError(f"sequence gap {seq} -> {s}; resyncing the book")
                    seq = s
                ch = msg.get("channel")
                for e in msg.get("events", []):
                    if ch == "market_trades":
                        self.on_trades(e.get("trades", []), snapshot=e.get("type") == "snapshot")
                    elif ch == "l2_data":
                        self.on_book(e)
                now = time.time()
                if now >= next_sample:
                    self.sample_books(now)
                    self.flush(now)
                    next_sample = now + BOOK_SAMPLE_SECONDS

    async def run(self):
        backoff = 1
        while True:
            started = time.time()
            try:
                await self.run_once()
            except Exception as e:
                self.reconnects += 1
                self.db.execute("INSERT OR REPLACE INTO status VALUES ('reconnects', ?)", (str(self.reconnects),))
                self.db.commit()
                if time.time() - started > 60:
                    backoff = 1
                log.warning("WebSocket dropped (%s); reconnecting in %ss", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    db = open_db()
    db.execute("INSERT OR IGNORE INTO status VALUES ('started', ?)", (str(time.time()),))
    db.commit()
    asyncio.run(Recorder(config.PRODUCT_IDS, db).run())


if __name__ == "__main__":
    main()
