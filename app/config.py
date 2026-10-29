"""All runtime settings, read from environment variables (see .env.example)."""
import os


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


def _s(name, default):
    return os.environ.get(name, default).strip()


DATA_DIR = _s("DATA_DIR", "/data")

# Market
PRODUCT_ID = _s("PRODUCT_ID", "BTC-USD")
GRANULARITY = _i("GRANULARITY", 3600)  # seconds per candle: 60, 300, 900, 1800, 3600, 7200, 21600, 86400

# Fees & execution costs (Coinbase Advanced Trade, lowest volume tier by default).
# Market orders always pay the taker fee.
FEE_SOURCE = _s("FEE_SOURCE", "auto")  # auto = use your real fee tier if API keys are set, else TAKER_FEE
TAKER_FEE = _f("TAKER_FEE", 0.012)  # 1.20%
SLIPPAGE_BPS = _f("SLIPPAGE_BPS", 5.0)  # extra adverse move on every fill (latency / impact), basis points
SPREAD_BPS = _f("SPREAD_BPS", 2.0)  # backtest-only: candles have no bid/ask, so assume this spread (live uses the real book)

# Strategy
POSITION_FRACTION = _f("POSITION_FRACTION", 1.0)  # share of available cash used per buy
BUY_THRESHOLD = _f("BUY_THRESHOLD", 0.2)  # network output above this -> buy (when flat)
SELL_THRESHOLD = _f("SELL_THRESHOLD", -0.2)  # network output below this -> sell (when holding)

# Training
TRAIN_DAYS = _f("TRAIN_DAYS", 60)
VALIDATION_DAYS = _f("VALIDATION_DAYS", 14)
GENERATIONS_PER_CYCLE = _i("GENERATIONS_PER_CYCLE", 100)
RETRAIN_INTERVAL_HOURS = _f("RETRAIN_INTERVAL_HOURS", 3)
POP_SIZE = _i("POP_SIZE", 150)
NUM_WORKERS = _i("NUM_WORKERS", 0)  # 0 = all CPU cores
FITNESS_CHUNKS = _i("FITNESS_CHUNKS", 4)  # score consistency across this many sub-periods
DRAWDOWN_PENALTY = _f("DRAWDOWN_PENALTY", 0.5)
INACTIVITY_PENALTY = _f("INACTIVITY_PENALTY", 0.25)

# Trading
TRADING_MODE = _s("TRADING_MODE", "paper").lower()  # paper | live
PAPER_START_USD = _f("PAPER_START_USD", 1000.0)
WEB_PORT = _i("WEB_PORT", 8080)

# Live trading (only used when TRADING_MODE=live)
COINBASE_API_KEY = _s("COINBASE_API_KEY", "")
COINBASE_API_SECRET = os.environ.get("COINBASE_API_SECRET", "").replace("\\n", "\n").strip()
LIVE_CONFIRM = _s("LIVE_CONFIRM", "")  # must equal I_UNDERSTAND_THIS_USES_REAL_MONEY
LIVE_MAX_USD = _f("LIVE_MAX_USD", 50.0)  # the bot never manages more than this much USD
HALT_DRAWDOWN_PCT = _f("HALT_DRAWDOWN_PCT", 25.0)  # stop trading if equity falls this far from its peak

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

CHAMPION_PATH = os.path.join(DATA_DIR, "champion.pkl")
TRAINER_DB = os.path.join(DATA_DIR, "trainer.db")
TRADER_DB = os.path.join(DATA_DIR, "trader.db")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "population.ckpt")
KILL_SWITCH_PATH = os.path.join(DATA_DIR, "STOP")
NEAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "neat_config.ini")


def has_api_keys():
    return bool(COINBASE_API_KEY and COINBASE_API_SECRET)
