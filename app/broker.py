"""Order execution. PaperBroker fills against the *live* Coinbase order book with
the same fee model used in training; CoinbaseBroker places real market orders."""
import logging
import math
import time
import uuid

from . import coinbase_data, config
from .sim import buy_with_cash, sell_base

log = logging.getLogger(__name__)


def floor_to(x, inc):
    return math.floor(x / inc + 1e-9) * inc


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

    def __init__(self, taker_fee):
        self.taker_fee = taker_fee
        self.product = coinbase_data.get_product(config.PRODUCT_ID)

    def mark_price(self):
        """Best bid after slippage: what selling would actually get, before the fee."""
        bids, _ = coinbase_data.get_order_book(config.PRODUCT_ID, limit=1)
        return bids[0][0] * (1 - config.SLIPPAGE_BPS / 10_000)

    def buy(self, cash):
        _, asks = coinbase_data.get_order_book(config.PRODUCT_ID)
        notional = cash / (1 + self.taker_fee)
        if notional < self.product["quote_min_size"]:
            return None
        price = walk_book(asks, quote=notional) * (1 + config.SLIPPAGE_BPS / 10_000)
        qty, notional, fee = buy_with_cash(cash, price, self.taker_fee)
        qty = floor_to(qty, self.product["base_increment"])
        return {"side": "buy", "qty": qty, "price": price, "fee": fee, "cash_delta": -cash}

    def sell(self, qty):
        qty = floor_to(qty, self.product["base_increment"])
        if qty < self.product["base_min_size"]:
            return None
        bids, _ = coinbase_data.get_order_book(config.PRODUCT_ID)
        price = walk_book(bids, base=qty) * (1 - config.SLIPPAGE_BPS / 10_000)
        got, _, fee = sell_base(qty, price, self.taker_fee)
        return {"side": "sell", "qty": qty, "price": price, "fee": fee, "cash_delta": got}


class CoinbaseBroker(PaperBroker):
    """Real money. Only trades the budget the bot tracks itself (capped by LIVE_MAX_USD)."""
    live = True

    def __init__(self, taker_fee):
        super().__init__(taker_fee)
        from .fees import coinbase_client
        self.client = coinbase_client()

    def _wait_fill(self, order_id):
        for _ in range(30):
            o = self.client.get_order(order_id).to_dict()["order"]
            if o["status"] in ("FILLED", "CANCELLED", "EXPIRED", "FAILED"):
                return o
            time.sleep(1)
        return o

    def _place(self, side, **size):
        fn = self.client.market_order_buy if side == "buy" else self.client.market_order_sell
        r = fn(client_order_id=str(uuid.uuid4()), product_id=config.PRODUCT_ID, **size).to_dict()
        if not r.get("success"):
            log.error("Coinbase rejected %s order: %s", side, r.get("error_response"))
            return None
        o = self._wait_fill(r["success_response"]["order_id"])
        filled = float(o.get("filled_size") or 0)
        if filled <= 0:
            log.error("Order %s not filled: %s", o.get("order_id"), o.get("status"))
            return None
        fee = float(o.get("total_fees") or 0)
        value = float(o.get("filled_value") or 0)
        price = float(o.get("average_filled_price") or value / filled)
        cash_delta = -(value + fee) if side == "buy" else value - fee
        return {"side": side, "qty": filled, "price": price, "fee": fee,
                "cash_delta": cash_delta, "order_id": o.get("order_id")}

    def buy(self, cash):
        quote = floor_to(cash / (1 + self.taker_fee), self.product["quote_increment"])
        if quote < self.product["quote_min_size"]:
            return None
        return self._place("buy", quote_size=f"{quote:.8f}".rstrip("0").rstrip("."))

    def sell(self, qty):
        qty = floor_to(qty, self.product["base_increment"])
        if qty < self.product["base_min_size"]:
            return None
        return self._place("sell", base_size=f"{qty:.10f}".rstrip("0").rstrip("."))
