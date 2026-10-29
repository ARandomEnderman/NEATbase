"""All runtime settings. Each one is read from, in order of precedence:
1. CONFIG_DIR/settings.env on the host (optional, KEY=VALUE lines)
2. environment variables (compose.yaml / the ZimaOS app settings screen)
3. the defaults below
Coinbase API keys are read from CONFIG_DIR/cdp_api_key.json (see README)."""
import json
import os

CONFIG_DIR = os.environ.get("CONFIG_DIR", "/config")
SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.env")
API_KEY_FILE = os.path.join(CONFIG_DIR, "cdp_api_key.json")


def _read_settings_file(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                out[k.strip()] = v
    except FileNotFoundError:
        pass
    return out


_FILE = _read_settings_file(SETTINGS_FILE)


def _get(name, default):
    return _FILE.get(name, os.environ.get(name, default))


def _f(name, default):
    return float(_get(name, default))


def _i(name, default):
    return int(_get(name, default))


def _s(name, default):
    return str(_get(name, default)).strip()


DATA_DIR = _s("DATA_DIR", "/data")

# Market
# One network trades every coin in this list, each with its own share of the money.
PRODUCT_IDS = [p.strip().upper() for p in _s(
    "PRODUCT_IDS", _s("PRODUCT_ID", "BTC-USD,ETH-USD,SOL-USD")).split(",") if p.strip()]
GRANULARITY = _i("GRANULARITY", 3600)  # seconds per candle: 60, 300, 900, 1800, 3600, 7200, 21600, 86400

# Fees & execution costs (Coinbase Advanced Trade, lowest volume tier by default).
# Market orders always pay the taker fee.
FEE_SOURCE = _s("FEE_SOURCE", "auto")  # auto = use your real fee tier if API keys are set, else TAKER_FEE
ORDER_TYPE = _s("ORDER_TYPE", "maker").lower()  # maker = post-only limit orders, taker = market orders
TAKER_FEE = _f("TAKER_FEE", 0.012)  # 1.20%
MAKER_FEE = _f("MAKER_FEE", 0.006)  # 0.60%
MAKER_SELL_FALLBACK_BARS = _i("MAKER_SELL_FALLBACK_BARS", 2)  # unfilled limit sells -> market sell after this many bars
SLIPPAGE_BPS = _f("SLIPPAGE_BPS", 5.0)  # extra adverse move on every fill (latency / impact), basis points
SPREAD_BPS = _f("SPREAD_BPS", 2.0)  # backtest-only: candles have no bid/ask, so assume this spread (live uses the real book)

# Strategy
# Position size is set by signal strength: a buy signal just over BUY_THRESHOLD invests
# 1/SIZE_STEPS of MAX_POSITION_FRACTION of that coin's pot, the strongest signal the full
# MAX_POSITION_FRACTION. The rest of the pot always stays in cash.
MAX_POSITION_FRACTION = _f("MAX_POSITION_FRACTION", 0.5)
SIZE_STEPS = _i("SIZE_STEPS", 4)
BUY_THRESHOLD = _f("BUY_THRESHOLD", 0.2)  # network output above this -> buy (when flat)
SELL_THRESHOLD = _f("SELL_THRESHOLD", -0.2)  # network output below this -> sell (when holding)

# Training
TRAIN_DAYS = _f("TRAIN_DAYS", 90)
VALIDATION_DAYS = _f("VALIDATION_DAYS", 14)
GENERATIONS_PER_CYCLE = _i("GENERATIONS_PER_CYCLE", 100)
RETRAIN_INTERVAL_HOURS = _f("RETRAIN_INTERVAL_HOURS", 3)
POP_SIZE = _i("POP_SIZE", 150)
NUM_WORKERS = _i("NUM_WORKERS", 0)  # 0 = all CPU cores
FITNESS_CHUNKS = _i("FITNESS_CHUNKS", 4)  # score consistency across this many sub-periods
# alpha: score = return beyond simply holding the same average amount of the coin, so
# riding a bull market scores ~0 and only timing counts. return: raw return.
FITNESS_MODE = _s("FITNESS_MODE", "alpha").lower()
DRAWDOWN_PENALTY = _f("DRAWDOWN_PENALTY", 0.5)
INACTIVITY_PENALTY = _f("INACTIVITY_PENALTY", 0.25)
# A candidate that beats the champion on validation must then also beat it on data
# that didn't exist yet when it was trained, before it may trade.
PROBATION_DAYS = _f("PROBATION_DAYS", 7)
MAX_CHALLENGERS = _i("MAX_CHALLENGERS", 6)

# Trading
TRADING_MODE = _s("TRADING_MODE", "paper").lower()  # paper | live
PAPER_START_USD = _f("PAPER_START_USD", 1000.0)
WEB_PORT = _i("WEB_PORT", 8080)

# Live trading (only used when TRADING_MODE=live)
COINBASE_API_KEY = _s("COINBASE_API_KEY", "")
COINBASE_API_SECRET = _s("COINBASE_API_SECRET", "").replace("\\n", "\n")
API_KEY_SOURCE = "settings" if COINBASE_API_KEY else None
if os.path.exists(API_KEY_FILE):
    # The JSON file you download when creating a key on portal.cdp.coinbase.com
    with open(API_KEY_FILE) as f:
        _key = json.load(f)
    COINBASE_API_KEY = (_key.get("name") or _key.get("id") or "").strip()
    COINBASE_API_SECRET = (_key.get("privateKey") or "").strip()
    API_KEY_SOURCE = "cdp_api_key.json"
LIVE_CONFIRM = _s("LIVE_CONFIRM", "")  # must equal I_UNDERSTAND_THIS_USES_REAL_MONEY
LIVE_MAX_USD = _f("LIVE_MAX_USD", 50.0)  # the bot never manages more than this much USD
HALT_DRAWDOWN_PCT = _f("HALT_DRAWDOWN_PCT", 25.0)  # stop trading if equity falls this far from its peak

# Web UI: only these source networks may connect (LAN, Tailscale, loopback, link-local)
ALLOWED_NETWORKS = [n.strip() for n in _s(
    "ALLOWED_NETWORKS",
    "127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,100.64.0.0/10,"
    "::1/128,fc00::/7,fe80::/10").split(",") if n.strip()]

GRANULARITY_NAMES = {
    60: "ONE_MINUTE",
    300: "FIVE_MINUTE",
    900: "FIFTEEN_MINUTE",
    1800: "THIRTY_MINUTE",
    3600: "ONE_HOUR",
    7200: "TWO_HOUR",
    21600: "SIX_HOUR",
    86400: "ONE_DAY",
}
if GRANULARITY not in GRANULARITY_NAMES:
    raise SystemExit(f"GRANULARITY must be one of {sorted(GRANULARITY_NAMES)}")
if FITNESS_MODE not in ("alpha", "return"):
    raise SystemExit("FITNESS_MODE must be alpha or return")
if ORDER_TYPE not in ("maker", "taker"):
    raise SystemExit("ORDER_TYPE must be maker or taker")
if not PRODUCT_IDS:
    raise SystemExit("PRODUCT_IDS is empty")

CHAMPION_PATH = os.path.join(DATA_DIR, "champion.pkl")
TRAINER_DB = os.path.join(DATA_DIR, "trainer.db")
TRADER_DB = os.path.join(DATA_DIR, "trader.db")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "population.ckpt")
CHALLENGERS_DIR = os.path.join(DATA_DIR, "challengers")
KILL_SWITCH_PATH = os.path.join(DATA_DIR, "STOP")
AUTH_PATH = os.path.join(DATA_DIR, "auth.json")
SECRET_KEY_PATH = os.path.join(DATA_DIR, "web_secret.key")
NEAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "neat_config.ini")


def has_api_keys():
    return bool(COINBASE_API_KEY and COINBASE_API_SECRET)
