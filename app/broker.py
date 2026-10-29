"""Order execution. PaperBroker fills against the *live* Coinbase order book and candles
using the same rules as training (see sim.py); CoinbaseBroker places real orders.

A fill is a dict: side, qty, price, fee, cash_delta (+ order_id when live).
A pending maker order is a dict: side, price, amount (cash for buys, base for sells),
bar (start time of the candle it was placed at the close of), order_id (live only).
"""
import logging
import math
import time
import uuid

from . import coinbase_data, config
from .sim import buy_with_cash, limit_fills, sell_base

log = logging.getLogger(__name__)


def floor_to(x, inc):
    return math.floor(x / inc + 1e-9) * inc


def ceil_to(x, inc):
    return math.ceil(x / inc - 1e-9) * inc


def fmt(x):
    return f"{x:.10f}".rstrip("0").rstrip(".")


def walk_book(levels, quote=None, base=None):
    """Fill `quote` dollars or `base` units against book levels. Returns average price."""
    got_base = got_quote = 0.0
    for price, size in levels:
        if quote is not None:
            take = min(size, (quote - got_quote) / price)
        else:
            take = min(size, base - got_base)
        got_base += take
        got_quote += take * price
        if (quote is not None and got_quote >= quote - 1e-9) or (base is not None and got_base >= base - 1e-12):
            return got_quote / got_base
    # Book shallower than our order: assume the rest fills 1% worse than the last level.
    last = levels[-1][0] * (1.01 if quote is not None else 0.99)
    rest_base = (quote - got_quote) / last if quote is not None else base - got_base
    return (got_quote + rest_base * last) / (got_base + rest_base)


class PaperBroker:
    live = False

    def __init__(self, taker_fee, maker_fee):
        self.taker_fee, self.maker_fee = taker_fee, maker_fee
        self.products = {pid: coinbase_data.get_product(pid) for pid in config.PRODUCT_IDS}

    def mark_price(self, pid):
        """Best bid after slippage: what a market sell would get, before the fee."""
        bids, _ = coinbase_data.get_order_book(pid, limit=1)
        return bids[0][0] * (1 - config.SLIPPAGE_BPS / 10_000)

    # ---- market (taker) orders ----
    def buy(self, pid, cash):
        p = self.products[pid]
        _, asks = coinbase_data.get_order_book(pid)
        if cash / (1 + self.taker_fee) < p["quote_min_size"]:
            return None
        price = walk_book(asks, quote=cash / (1 + self.taker_fee)) * (1 + config.SLIPPAGE_BPS / 10_000)
        qty, _, fee = buy_with_cash(cash, price, self.taker_fee)
        return {"side": "buy", "qty": floor_to(qty, p["base_increment"]), "price": price, "fee": fee,
                "cash_delta": -cash}

    def sell(self, pid, qty):
        p = self.products[pid]
        qty = floor_to(qty, p["base_increment"])
        if qty < p["base_min_size"]:
            return None
        bids, _ = coinbase_data.get_order_book(pid)
        price = walk_book(bids, base=qty) * (1 - config.SLIPPAGE_BPS / 10_000)
        got, _, fee = sell_base(qty, price, self.taker_fee)
        return {"side": "sell", "qty": qty, "price": price, "fee": fee, "cash_delta": got}

    # ---- post-only limit (maker) orders ----
    def place_limit(self, pid, side, amount, bar):
        """Rest an order at the best bid (buy) / best ask (sell). `amount` is cash or base."""
        p = self.products[pid]
        bids, asks = coinbase_data.get_order_book(pid, limit=1)
        price = bids[0][0] if side == "buy" else asks[0][0]
        if side == "buy" and amount / (1 + self.maker_fee) < p["quote_min_size"]:
            return None
        if side == "sell" and floor_to(amount, p["base_increment"]) < p["base_min_size"]:
            return None
        return {"side": side, "price": price, "amount": amount, "bar": bar}

    def resolve_limit(self, pid, order, candles):
        """Paper rule, same as training: filled only if the next candle traded through."""
        nxt = next((c for c in candles if c[0] == order["bar"] + config.GRANULARITY), None)
        if nxt is None or not limit_fills(order["side"], order["price"], nxt):
            return None
        p = self.products[pid]
        if order["side"] == "buy":
            qty, _, fee = buy_with_cash(order["amount"], order["price"], self.maker_fee)
            return {"side": "buy", "qty": floor_to(qty, p["base_increment"]), "price": order["price"],
                    "fee": fee, "cash_delta": -order["amount"]}
        qty = floor_to(order["amount"], p["base_increment"])
        got, _, fee = sell_base(qty, order["price"], self.maker_fee)
        return {"side": "sell", "qty": qty, "price": order["price"], "fee": fee, "cash_delta": got}


class CoinbaseBroker(PaperBroker):
    """Real money. Only trades the budget the bot tracks itself (capped by LIVE_MAX_USD)."""
    live = True

    def __init__(self, taker_fee, maker_fee):
        super().__init__(taker_fee, maker_fee)
        from .fees import coinbase_client
        self.client = coinbase_client()

    def _order(self, order_id):
        return self.client.get_order(order_id).to_dict()["order"]

    def _fill_from(self, o, side):
        filled = float(o.get("filled_size") or 0)
        if filled <= 0:
            return None
        fee = float(o.get("total_fees") or 0)
        value = float(o.get("filled_value") or 0)
        price = float(o.get("average_filled_price") or value / filled)
        return {"side": side, "qty": filled, "price": price, "fee": fee,
                "cash_delta": -(value + fee) if side == "buy" else value - fee,
                "order_id": o.get("order_id")}

    def _submit(self, fn, **kw):
        r = fn(client_order_id=str(uuid.uuid4()), **kw).to_dict()
        if not r.get("success"):
            log.error("Coinbase rejected order %s: %s", kw, r.get("error_response"))
            return None
        return r["success_response"]["order_id"]

    def _market(self, side, pid, **size):
        fn = self.client.market_order_buy if side == "buy" else self.client.market_order_sell
        order_id = self._submit(fn, product_id=pid, **size)
        if not order_id:
            return None
        for _ in range(30):
            o = self._order(order_id)
            if o["status"] in ("FILLED", "CANCELLED", "EXPIRED", "FAILED"):
                break
            time.sleep(1)
        return self._fill_from(o, side)

    def buy(self, pid, cash):
        p = self.products[pid]
        quote = floor_to(cash / (1 + self.taker_fee), p["quote_increment"])
        if quote < p["quote_min_size"]:
            return None
        return self._market("buy", pid, quote_size=fmt(quote))

    def sell(self, pid, qty):
        p = self.products[pid]
        qty = floor_to(qty, p["base_increment"])
        if qty < p["base_min_size"]:
            return None
        return self._market("sell", pid, base_size=fmt(qty))

    def place_limit(self, pid, side, amount, bar):
        order = super().place_limit(pid, side, amount, bar)
        if order is None:
            return None
        p = self.products[pid]
        inc = p.get("price_increment") or p["quote_increment"]
        price = floor_to(order["price"], inc) if side == "buy" else ceil_to(order["price"], inc)
        if side == "buy":
            base = floor_to(amount / (1 + self.maker_fee) / price, p["base_increment"])
            fn = self.client.limit_order_gtc_buy
        else:
            base = floor_to(amount, p["base_increment"])
            fn = self.client.limit_order_gtc_sell
        if base < p["base_min_size"]:
            return None
        order_id = self._submit(fn, product_id=pid, base_size=fmt(base), limit_price=fmt(price), post_only=True)
        if not order_id:
            return None  # e.g. post-only would have crossed the book: treat as unfilled
        return {**order, "price": price, "order_id": order_id}

    def resolve_limit(self, pid, order, candles):
        """Cancel whatever is left of the order, then book whatever actually filled."""
        o = self._order(order["order_id"])
        if o["status"] not in ("FILLED", "CANCELLED", "EXPIRED", "FAILED"):
            self.client.cancel_orders(order_ids=[order["order_id"]])
            for _ in range(10):
                o = self._order(order["order_id"])
                if o["status"] in ("FILLED", "CANCELLED", "EXPIRED", "FAILED"):
                    break
                time.sleep(1)
        return self._fill_from(o, order["side"])
