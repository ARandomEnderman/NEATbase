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
- **Web UI** on port 8080: shows equity vs buy & hold, fees paid, win rate, drawdown,
  trades, fitness per generation, and each candidate's validation curve. It has a
  Pause/Resume switch and is password protected. Only local networks can reach it
  (see below).

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

Open `http://<zima-ip>:8080` (or `http://<tailscale-name>:8080`). **The first visit asks
you to create the password.** Do this right after deploying. The first training cycle
takes a minute or two. After that the trader makes one decision per candle (hourly by
default).

### Option B: install from the ZimaOS dashboard

Pushing to `main` runs `.github/workflows/docker.yml`. It publishes a multi-arch image
(amd64 + arm64) to `ghcr.io/arandomenderman/neatbase:latest`. If the repo is private,
make the package public under GitHub → Packages → neatbase → Package settings, or run
`docker login ghcr.io` on the server. Then in ZimaOS go to **App Store → Custom Install →
Import** and paste `docker-compose.yml`. You can edit the settings in the app's settings
screen.

### Folders on the host

| Host folder | In container | Contents |
|---|---|---|
| `/DATA/AppData/neatbase/config` | `/config` (read-only) | **your files:** `cdp_api_key.json`, `settings.env` |
| `/DATA/AppData/neatbase/data` | `/data` | candle cache, champion, population checkpoint, trades, web password hash |

Set `CONFIG_PATH` / `DATA_PATH` to use other folders on non-ZimaOS hosts.

## Config folder: API keys and settings

Your secrets never go in `docker-compose.yml`. Put them in the config folder on the
host:

```sh
mkdir -p /DATA/AppData/neatbase/config
cp ~/Downloads/cdp_api_key.json /DATA/AppData/neatbase/config/   # the file Coinbase gave you
chmod 600 /DATA/AppData/neatbase/config/cdp_api_key.json
docker compose restart
```

- **`cdp_api_key.json`** is the JSON file you download when you create a key at
  <https://portal.cdp.coinbase.com/>. It contains `"name"` (or `"id"`) and
  `"privateKey"`. Both ECDSA and Ed25519 keys work. See `config/cdp_api_key.json.example`.
  Give the key **View + Trade** permissions only, never Transfer.
- **`settings.env`** is optional. Any setting from the table below goes here as `KEY=value`
  lines, and it overrides the compose file and `.env`. See `config/settings.env.example`.

Both files are only read at startup, so run `docker compose restart` after editing.
When a key is loaded, the dashboard header shows its last 6 characters, and your
**real Coinbase fee tier** is used for training and paper trading. A key alone never
enables live trading.

## Web UI access

- **Local networks only.** Requests are refused (HTTP 403) unless they come from LAN
  (`10/8`, `172.16/12`, `192.168/16`), Tailscale (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`),
  localhost or link-local. The check uses the connecting IP address and ignores
  `X-Forwarded-For`-style headers. You can change the list with `ALLOWED_NETWORKS`.
- **Password set on first login.** It is stored as a salted scrypt hash in
  `data/auth.json`. Five wrong attempts lock that IP out for 5 minutes. Changing the
  password (via the **Password** button) signs out all other browsers.
- **Forgot the password?** `rm /DATA/AppData/neatbase/data/auth.json`, then visit the page
  and set a new one.
- Don't port-forward 8080 on your router. Reach it remotely through Tailscale instead.
  Note that if you put a reverse proxy in front of it on the same host, every request
  will appear to come from the proxy's local IP.

## Useful commands

```sh
docker compose logs -f trainer           # watch evolution
docker compose logs -f trader            # every decision, fill and fee
touch /DATA/AppData/neatbase/data/STOP   # same as the Pause button: sell at the next candle, stop buying
rm /DATA/AppData/neatbase/data/STOP      # resume
docker compose down && rm /DATA/AppData/neatbase/data/trader.db*   # restart paper trading from scratch
rm /DATA/AppData/neatbase/data/{champion.pkl,population.ckpt*}     # restart evolution from scratch
```

## Settings

Every setting can go in the compose environment, in `.env`, or in
`config/settings.env` on the host (which wins). See `.env.example` for the full list with
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

1. Put your `cdp_api_key.json` in the config folder (see above). It needs **View** and
   **Trade** permissions only; never enable transfer/withdraw. If you can, restrict the
   key to your server's IP.
2. Create `/DATA/AppData/neatbase/config/settings.env` with:
   ```sh
   TRADING_MODE=live
   LIVE_CONFIRM=I_UNDERSTAND_THIS_USES_REAL_MONEY
   LIVE_MAX_USD=50          # the bot never manages more than this
   HALT_DRAWDOWN_PCT=25     # flatten and stop if equity drops this far from its peak
   ```
3. `docker compose restart`. The dashboard badge turns red and says **LIVE**.

In live mode the bot places market orders through the official `coinbase-advanced-py`
SDK. It records the real fill price and fees from Coinbase, and only trades its own
tracked budget (at most `LIVE_MAX_USD`), not your whole balance. Switching between
paper and live starts a fresh session. The Pause button / `STOP` file works in both
modes.

> This is an experiment, not financial advice. Automated trading can lose money quickly,
> and a strategy that worked in paper trading can fail live.

## Development

```sh
pip install -r requirements.txt pytest
python -m pytest tests
DATA_DIR=./data GENERATIONS_PER_CYCLE=5 python -m app.trainer
DATA_DIR=./data python -m app.trader
```
