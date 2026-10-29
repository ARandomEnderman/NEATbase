"""Public Coinbase Advanced Trade market data (no API key needed)."""
import logging
import sqlite3
import time

import requests

from . import config

log = logging.getLogger(__name__)

BASE_URL = "https://api.coinbase.com/api/v3/brokerage/market"
MAX_CANDLES_PER_REQUEST = 350

_session = requests.Session()
_session.headers["User-Agent"] = "neatbase/1.0"


def _get(path, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = _session.get(f"{BASE_URL}{path}", params=params, timeout=15)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** attempt
            log.warning("Coinbase request %s failed (%s), retrying in %ss", path, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Coinbase request {path} failed after {retries} attempts")


def get_product(product_id):
    p = _get(f"/products/{product_id}")
    return {
        "base_increment": float(p["base_increment"]),
        "quote_increment": float(p["quote_increment"]),
        "quote_min_size": float(p["quote_min_size"]),
        "base_min_size": float(p["base_min_size"]),
        "status": p.get("status"),
        "trading_disabled": p.get("trading_disabled", False),
    }


def get_order_book(product_id, limit=200):
    book = _get("/product_book", {"product_id": product_id, "limit": limit})["pricebook"]
    bids = [(float(l["price"]), float(l["size"])) for l in book["bids"]]
    asks = [(float(l["price"]), float(l["size"])) for l in book["asks"]]
    if not bids or not asks:
        raise RuntimeError(f"Empty order book for {product_id}")
    return bids, asks


def current_spread_bps(product_id):
    bids, asks = get_order_book(product_id, limit=1)
    mid = (bids[0][0] + asks[0][0]) / 2
    return (asks[0][0] - bids[0][0]) / mid * 10_000


def fetch_candles(product_id, granularity, start, end):
    """Closed candles with start in [start, end), oldest first, as tuples
    (start, open, high, low, close, volume)."""
    name = config.GRANULARITY_NAMES[granularity]
    now = int(time.time())
    end = min(end, now - now % granularity)  # never include the still-forming candle
    out = {}
    chunk_end = end
    while chunk_end > start:
        chunk_start = max(start, chunk_end - granularity * MAX_CANDLES_PER_REQUEST)
        data = _get(f"/products/{product_id}/candles", {
            "start": str(chunk_start),
            "end": str(chunk_end - 1),
            "granularity": name,
            "limit": MAX_CANDLES_PER_REQUEST,
        })
        for c in data.get("candles", []):
            t = int(c["start"])
            if start <= t < end:
                out[t] = (t, float(c["open"]), float(c["high"]), float(c["low"]),
                          float(c["close"]), float(c["volume"]))
        chunk_end = chunk_start
        time.sleep(0.12)  # stay well under the public rate limit
    return [out[t] for t in sorted(out)]


class CandleCache:
    """SQLite cache so retraining only downloads candles it doesn't have yet."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS candles (
            product TEXT, granularity INTEGER, start INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (product, granularity, start))""")
        self.db.commit()

    def get(self, product_id, granularity, start, end):
        now = int(time.time())
        end = min(end, now - now % granularity)
        row = self.db.execute(
            "SELECT MIN(start), MAX(start) FROM candles WHERE product=? AND granularity=?",
            (product_id, granularity)).fetchone()
        lo, hi = row
        ranges = []
        if lo is None:
            ranges.append((start, end))
        else:
            if start < lo:
                ranges.append((start, lo))
            if hi + granularity < end:
                ranges.append((hi + granularity, end))
        for a, b in ranges:
            log.info("Downloading %s candles %s -> %s", product_id,
                     time.strftime("%Y-%m-%d %H:%M", time.gmtime(a)),
                     time.strftime("%Y-%m-%d %H:%M", time.gmtime(b)))
            rows = fetch_candles(product_id, granularity, a, b)
            self.db.executemany(
                "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)",
                [(product_id, granularity) + r for r in rows])
            self.db.commit()
        return self.db.execute(
            "SELECT start, open, high, low, close, volume FROM candles "
            "WHERE product=? AND granularity=? AND start>=? AND start<? ORDER BY start",
            (product_id, granularity, start, end)).fetchall()
