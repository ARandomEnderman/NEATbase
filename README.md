# NEATbase

A small, self-hosted NEAT (NeuroEvolution of Augmenting Topologies) crypto trader for
Coinbase. It evolves neural networks on real Coinbase candles and **paper trades** the
best one on the live market. Every simulated fill pays Coinbase's fees. It runs as a
two-container Docker Compose app and is built for ZimaOS.

```
┌─────────────┐  champion.pkl  ┌──────────────┐   http://<zima-ip>:8080
│  trainer    │ ─────────────▶ │  trader      │ ──▶ dashboard
│ NEAT evolve │                │ paper / live │
└─────▲───────┘                └──────▲───────┘
      │ historical candles            │ closed candles + live order book
      └──────── Coinbase Advanced Trade public API (no key needed) ─────┘
```

## What it does

- **Trainer:** downloads the last `TRAIN_DAYS + VALIDATION_DAYS` of candles and caches
  them in SQLite. It then evolves a population of networks for `GENERATIONS_PER_CYCLE`
  generations on the training window. The best network is tested on the most recent
  `VALIDATION_DAYS`, which it never trained on. It is promoted to *champion* only if it
  beats the current champion on that same held-out data. The population is checkpointed
  and keeps evolving on fresh data every `RETRAIN_INTERVAL_HOURS`.
- **Trader:** at every candle close it feeds the champion the same 16 inputs used in
  training: volatility-normalised returns, RSI, EMA trend, volume, range, time of day, and
  position / unrealised P&L / bars held. It buys or sells against the **real live order
  book**. It also tracks a buy & hold benchmark that pays the same fees.
- **Dashboard** on port 8080: shows equity vs buy & hold, fees paid, win rate, drawdown,
  trades, fitness per generation, and each candidate's validation curve.

### Costs included in the simulation

Training and paper trading use the same cost model (`app/sim.py`). That way a network
can't learn a strategy that only works without fees.

| Cost | Training (backtest) | Paper trading (live) |
|---|---|---|
| Taker fee, charged on both buy and sell | `TAKER_FEE`, default 1.20% (Coinbase's lowest volume tier) | same |
| Spread | max(`SPREAD_BPS`, live spread when the cycle starts) | real bid/ask from the order book |
| Market impact | — | the order is walked through the order book levels |
| Slippage / latency | `SLIPPAGE_BPS` per fill | `SLIPPAGE_BPS` per fill |
| Order size limits | — | Coinbase `base_increment` / minimum order size |
| Equity valuation | liquidation value (bid − slippage − sell fee) | same |

If you add Coinbase API keys, both containers read your **real fee tier** from
Coinbase, even in paper mode.

A full round trip at the default tier costs about **2.5%**. Most short-term strategies
can't overcome that, which is why the default is 1-hour candles. Expect the bot to spend
a lot of time in cash. When that happens it's usually the right call, not a bug. Compare
it against the buy & hold line, not against zero.

## Deploy on ZimaOS

### Option A: build on the server (works right away)

SSH into ZimaOS, then:

```sh
git clone https://github.com/ARandomEnderman/NEATbase.git /DATA/AppData/neatbase-src
cd /DATA/AppData/neatbase-src
cp .env.example .env        # optional: edit settings
docker compose up -d --build
```

Open `http://<zima-ip>:8080`. The first training cycle takes a minute or two. After that
the trader makes one decision per candle (hourly by default).

### Option B: install from the ZimaOS dashboard

Pushing to `main` runs `.github/workflows/docker.yml`. It publishes a multi-arch image
(amd64 + arm64) to `ghcr.io/arandomenderman/neatbase:latest`. If the repo is private,
make the package public under GitHub → Packages → neatbase → Package settings, or run
`docker login ghcr.io` on the server. Then in ZimaOS go to **App Store → Custom Install →
Import** and paste `docker-compose.yml`. You can edit the settings in the app's settings
screen.

Data (candle cache, champion, population checkpoint, trades) is stored in
`/DATA/AppData/neatbase`. Set `DATA_PATH` to use a different folder on other hosts.

## Useful commands

```sh
docker compose logs -f trainer           # watch evolution
docker compose logs -f trader            # every decision, fill and fee
touch /DATA/AppData/neatbase/STOP        # kill switch: at the next candle, sell and stop buying
rm /DATA/AppData/neatbase/STOP           # resume
docker compose down && rm /DATA/AppData/neatbase/trader.db*   # restart paper trading from scratch
rm /DATA/AppData/neatbase/{champion.pkl,population.ckpt*}     # restart evolution from scratch
```

## Settings

All settings are environment variables. See `.env.example` for the full list with
comments. The most useful ones:

| Variable | Default | |
|---|---|---|
| `PRODUCT_ID` | `BTC-USD` | any Coinbase spot pair |
| `GRANULARITY` | `3600` | candle size in seconds (60, 300, 900, 1800, 3600, 7200, 21600, 86400) |
| `TAKER_FEE` | `0.012` | your fee tier (ignored when API keys are set) |
| `SLIPPAGE_BPS` / `SPREAD_BPS` | `5` / `2` | extra execution costs |
| `TRAIN_DAYS` / `VALIDATION_DAYS` | `60` / `14` | training and held-out windows |
| `POP_SIZE`, `GENERATIONS_PER_CYCLE` | `150`, `100` | evolution size |
| `TRAINER_CPUS`, `NUM_WORKERS` | `2`, `2` | how much of the server the trainer may use |
| `PAPER_START_USD` | `1000` | starting paper balance |

## Going live (real money)

Paper trade for a few weeks first, and only go live if the bot beats buy & hold *after
fees* over that period. A good backtest is not evidence; the validation column on the
dashboard exists because training results look much better than they really are.

1. Create a CDP API key at <https://portal.cdp.coinbase.com/> with **View** and **Trade**
   permissions only. Never enable transfer/withdraw permissions. If you can, restrict the
   key to your server's IP.
2. In `.env`:
   ```sh
   COINBASE_API_KEY=organizations/.../apiKeys/...
   COINBASE_API_SECRET=-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n
   TRADING_MODE=live
   LIVE_CONFIRM=I_UNDERSTAND_THIS_USES_REAL_MONEY
   LIVE_MAX_USD=50          # the bot never manages more than this
   HALT_DRAWDOWN_PCT=25     # flatten and stop if equity drops this far from its peak
   ```
3. `docker compose up -d`. The dashboard badge turns red and says **LIVE**.

In live mode the bot places market orders through the official `coinbase-advanced-py`
SDK. It records the real fill price and fees from Coinbase, and only trades its own
tracked budget (at most `LIVE_MAX_USD`), not your whole balance. Switching between
paper and live starts a fresh session. The `STOP` file kill switch works in both modes.

> This is an experiment, not financial advice. Automated trading can lose money quickly,
> and a strategy that worked in paper trading can fail live.

## Development

```sh
pip install -r requirements.txt pytest
python -m pytest tests
DATA_DIR=./data GENERATIONS_PER_CYCLE=5 python -m app.trainer
DATA_DIR=./data python -m app.trader
```
