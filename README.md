# NEATbase

A small, self-hosted NEAT (NeuroEvolution of Augmenting Topologies) crypto trader for
Coinbase. It evolves neural networks on real Coinbase candles and **paper trades** the
best one on the live market, across several coins at once. Every simulated fill pays
Coinbase's fees. It runs as a
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

- **Trainer:** downloads and caches candles for every coin in `PRODUCT_IDS` (BTC, ETH and
  SOL by default). It evolves **one shared network** on all of them over the last
  `TRAIN_DAYS` (90) days. Each coin's backtest has its own pot of money, and the network's
  score is the average across coins, which rewards strategies that work on more than
  one coin. The population is checkpointed and keeps evolving every
  `RETRAIN_INTERVAL_HOURS`.
- **Score = timing skill (alpha), not raw return.** If raw return were the goal, the
  easiest high score in a bull market would be to buy and never sell. So each period is
  scored as the bot's return **minus what simply holding its average % invested the
  whole time would have made**. Buy & hold scores about 0 (minus fees). Only being more
  invested before rises and less before drops scores positive. On top of that, the score
  subtracts unevenness across 4 sub-periods, 0.5 × the largest drop from peak, and 0.25
  if it never trades. `FITNESS_MODE=return` switches back to scoring raw return.
- **Two-stage promotion,** so lucky or overfitted networks don't get to trade:
  1. **Validation:** the cycle's best network must beat the champion on the last
     `VALIDATION_DAYS` (14), which it never trained on. If it does, it becomes a
     *challenger*. At most one challenger is kept per day (the best one).
  2. **Probation:** a challenger is promoted only if it also beats the champion over the
     next `PROBATION_DAYS` (7) of data that **didn't exist yet** when it was created.
     Each challenger gets exactly one forward test and is discarded if it loses, so no
     amount of retrying lets luck through.
  The very first network has nothing to compete against, so it becomes champion
  straight away; every later one goes through both stages.
- **Trader:** at every candle close it feeds the champion 20 inputs for each coin:
  - returns over 1/4/16/64 candles and over one week
  - RSI
  - EMA20/50 trend and the distance from EMA200
  - volatility regime
  - volume
  - position in its 20-candle and one-week ranges
  - candle body
  - time of day
  - the position (how big it is relative to the cap, P&L after fees, how long it's been held)
  - **its own previous output**, a one-step memory

  Each coin trades its own equal share of the money.
- **Position sizing:** the strength of the buy signal sets the size. A signal just over
  `BUY_THRESHOLD` invests ¼ of the maximum and the strongest signal invests the full
  maximum, in `SIZE_STEPS` (4) steps. The maximum is `MAX_POSITION_FRACTION` (0.5) of
  that coin's pot, so **at most half your money is ever in crypto**, even when every
  coin is bought at once. The size is set when buying, and a sell always sells the
  whole position. Training uses exactly the same rules, so the network learns when a
  bigger bet is worth it.
- **Benchmark:** a buy & hold line that pays the same fees. Note that buy & hold is
  100% invested while the bot is at most 50%, so in a rising market the bot can trail
  it even with good timing, and in a falling market it can beat it just by holding
  less.
- **Web UI** on port 8080 shows:
  - a price chart per coin with every buy ▲ and sell ▼ marked (size = dollars traded,
    hover for amount, fee and P&L)
  - equity vs buy & hold, fees, win rate and drawdown
  - the probation queue, training fitness and validation curves

  It has a Pause/Resume switch, is password protected, and only local networks can
  reach it (see below).

### Costs included in the simulation

Training and paper trading use the same cost model (`app/sim.py`). That way a network
can't learn a strategy that only works without fees.

**Orders.** By default (`ORDER_TYPE=maker`) the bot places post-only limit orders at the
best bid (to buy) or best ask (to sell) when a candle closes. These pay the **maker fee:
0.60%** instead of the 1.20% taker fee. An order counts as filled only if the **next**
candle trades *through* the limit price; just touching it is not enough. Unfilled
orders are cancelled. A sell that stays unfilled for `MAKER_SELL_FALLBACK_BARS` (2)
candles in a row becomes a market sell, so the bot can't get stuck in a falling coin.
The Pause button and the drawdown halt always use market orders. Training, paper
trading and live trading all follow these same rules. Set `ORDER_TYPE=taker` to only
use market orders.

| Cost | Training (backtest) | Paper trading (live) |
|---|---|---|
| Maker fee (limit orders) | `MAKER_FEE`, default 0.60% | same; fill needs a trade through the limit on the next candle |
| Taker fee (market orders, sell fallback) | `TAKER_FEE`, default 1.20% (Coinbase's lowest volume tier) | same |
| Spread | max(`SPREAD_BPS`, live spread when the cycle starts) | real bid/ask from the order book |
| Market impact | — | the order is walked through the order book levels |
| Slippage / latency | `SLIPPAGE_BPS` per fill | `SLIPPAGE_BPS` per fill |
| Order size limits | — | Coinbase `base_increment` / minimum order size |
| Equity valuation | liquidation value (bid − slippage − sell fee) | same |

If you add Coinbase API keys, both containers read your **real fee tier** from
Coinbase, even in paper mode.

A round trip costs about **1.2%** with maker orders and about **2.5%** with market
orders. Most short-term strategies can't overcome that, which is why the default is
1-hour candles. Expect the bot to spend
a lot of time in cash. When that happens it's usually the right call, not a bug. Compare
it against the buy & hold line, not against zero.

## Deploy on ZimaOS

`compose.yaml` is a complete ZimaOS app. There is nothing to build or clone on the
server.

1. **One-time: make the image public.** GitHub Actions publishes the image to
   `ghcr.io/arandomenderman/neatbase` on every push. Because this repo is private, the
   image starts out private and ZimaOS can't pull it. On GitHub, open your profile →
   **Packages** → **neatbase** → **Package settings** → **Change visibility** →
   **Public**. (The image contains this app's code but none of your keys or data.)
2. In the ZimaOS dashboard: **App Store** → **+** (top right) → **Install a customized
   app** → **Import** → upload `compose.yaml` → **Submit** → **Install**.
3. Open the **NEATbase** tile, or go to `http://<zima-ip>:8080` /
   `http://<tailscale-name>:8080`. **The first visit asks you to create the password.**
   Do this right after installing.

The first training cycle takes a minute or two. After that the trader makes one decision
per candle (hourly by default). To update, use **Check then update** in the app's menu
(it pulls `:latest`).

Not on ZimaOS? Build the image locally and use the same file:
`docker build -t ghcr.io/arandomenderman/neatbase:latest . && docker compose up -d`
(change the `/DATA/AppData/...` paths in `compose.yaml` first).

### Folders on the host

| Host folder | In container | Contents |
|---|---|---|
| `/DATA/AppData/neatbase/config` | `/config` (read-only) | **your files:** `cdp_api_key.json`, `settings.env` |
| `/DATA/AppData/neatbase/data` | `/data` | candle cache, champion, population checkpoint, trades, web password hash |

## Config folder: API keys and settings

Your secrets never go in `compose.yaml`. Put them in the config folder on the
host:

```sh
mkdir -p /DATA/AppData/neatbase/config
cp ~/Downloads/cdp_api_key.json /DATA/AppData/neatbase/config/   # the file Coinbase gave you
chmod 600 /DATA/AppData/neatbase/config/cdp_api_key.json
docker restart neatbase-trainer neatbase-trader
```

- **`cdp_api_key.json`** is the JSON file you download when you create a key at
  <https://portal.cdp.coinbase.com/>. It contains `"name"` (or `"id"`) and
  `"privateKey"`. Both ECDSA and Ed25519 keys work. See `config/cdp_api_key.json.example`.
  Give the key **View + Trade** permissions only, never Transfer.
- **`settings.env`** is optional. Any setting from the table below goes here as `KEY=value`
  lines, and it overrides the app's environment variables. See `config/settings.env.example`.

Both files are only read at startup, so restart the app after editing (ZimaOS app menu →
Restart, or `docker restart neatbase-trainer neatbase-trader`).
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
docker logs -f neatbase-trainer          # watch evolution
docker logs -f neatbase-trader           # every decision, fill and fee
touch /DATA/AppData/neatbase/data/STOP   # same as the Pause button: sell at the next candle, stop buying
rm /DATA/AppData/neatbase/data/STOP      # resume
docker stop neatbase-trader && rm /DATA/AppData/neatbase/data/trader.db* && docker start neatbase-trader  # restart paper trading
rm /DATA/AppData/neatbase/data/{champion.pkl,population.ckpt*}     # restart evolution from scratch
```

## Settings

Change settings either in the ZimaOS app settings screen (environment variables) or in
`config/settings.env` on the host, which wins over the app settings.
`config/settings.env.example` lists all of them with comments. The most useful ones:

| Variable | Default | |
|---|---|---|
| `PRODUCT_IDS` | `BTC-USD,ETH-USD,SOL-USD` | coins to trade, one shared network, equal pot each |
| `ORDER_TYPE` | `maker` | `maker` = limit orders at 0.60%, `taker` = market orders at 1.20% |
| `MAX_POSITION_FRACTION` | `0.5` | most of each coin's pot ever invested; 0.5 = never more than half your money in crypto |
| `SIZE_STEPS` | `4` | buy size steps between a weak signal (¼ of max) and the strongest (full max) |
| `GRANULARITY` | `3600` | candle size in seconds (60, 300, 900, 1800, 3600, 7200, 21600, 86400) |
| `TAKER_FEE` / `MAKER_FEE` | `0.012` / `0.006` | your fee tier (ignored when an API key is loaded) |
| `SLIPPAGE_BPS` / `SPREAD_BPS` | `5` / `2` | extra execution costs |
| `TRAIN_DAYS` / `VALIDATION_DAYS` | `90` / `14` | training and held-out windows |
| `PROBATION_DAYS` | `7` | forward test on unseen data before a new network may trade |
| `POP_SIZE`, `GENERATIONS_PER_CYCLE` | `150`, `30` | evolution size; 30 generations overfit less than 100 in walk-forward tests |
| `NUM_WORKERS` | `0` | CPU cores for training, `0` = all. The trainer runs at low priority (`nice 10`), so other apps go first. |
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
3. Restart the app (ZimaOS app menu → Restart, or `docker restart neatbase-trader`). The
   dashboard badge turns red and says **LIVE**.

In live mode the bot places market orders through the official `coinbase-advanced-py`
SDK. It records the real fill price and fees from Coinbase, and only trades its own
tracked budget (at most `LIVE_MAX_USD`), not your whole balance. Switching between
paper and live starts a fresh session. The Pause button / `STOP` file works in both
modes.

> This is an experiment, not financial advice. Automated trading can lose money quickly,
> and a strategy that worked in paper trading can fail live.

## Comparing settings (walk-forward test)

`app/compare.py` measures whether a setting helps on data that training never saw. It
uses the bot's own fees, order rules and sizing. It first holds back the last 30 days
(the *lockbox*). Before that it lays out several 30-day test months. For each month,
fresh populations are trained only on the 90 days before it, then scored on that month.
The score is **unseen alpha**: return beyond holding the same average amount.

```sh
docker run --rm -v neatbase-compare:/data ghcr.io/arandomenderman/neatbase:latest \
    python -m app.compare --folds 6 --runs 2
```

Options:
- `--granularities 3600,21600`: compare candle sizes
- `--folds` / `--fold-days`: number and length of test months (default 6 × 30 days)
- `--runs`: independent trainings per month
- `--ensemble N`: N populations that average their outputs
- `--generations`, `--pop`, `--train-days`, `--workers`

You can also pass any bot setting with `-e`, for example `-e FEATURES=cross`,
`-e ORDER_TYPE=taker` or `-e COMPLEXITY_PENALTY=0.3`.

Run `--final` **once**, after you've finished trying settings. It scores the lockbox. If
the lockbox result is much worse than the test months, your settings were tuned to
those months rather than learning something general.

### What the tests found so far

These are mean unseen alpha over 6 test months (Apr–Sep 2026), BTC/ETH/SOL, 1h candles
unless noted. "Months > 0" counts the months with positive alpha.

| Setting | Mean alpha | Months > 0 |
|---|---|---|
| default (100 generations at the time) | −0.80 | 2/6 |
| **30 generations** (now the default) | **−0.23** | 2/6 |
| cross-coin inputs | −1.06 | 3/6 |
| 180 training days | −0.96 | 1/6 |
| complexity penalty 0.3 | −0.50 | 0/6 |
| 6h candles | −0.99 | 2/6 |
| ensemble of 5 | −0.30 | 0/6 |
| 0.25% / 0.5% fees | −0.94 | 2/6 |
| **no fees at all** | **+0.53** | 4/6 |
| simple trend rules (no NEAT) | −2 to −10 | |
| lockbox (Sep 29 – Oct 29), `--final` | −0.94 | 0/1 |

So far nothing shows timing skill after fees, and even without fees the signal is only
about +0.5 per month. The bot mostly limits losses to about its fees. The forward-test
probation keeps networks without skill from taking over.

## Order-flow recorder (new data for future training)

The tests above suggest that hourly candles barely predict the next hour. Coinbase's
order flow and order book carry different information: who is buying or selling
aggressively, and how much is waiting on each side. Coinbase keeps no history of
these, so the `recorder` service records them from now on. It uses Coinbase's public
WebSocket (no key needed) and stores one row per coin per minute in `data/flow.db`
(a few MB per month).

Each minute it stores:
- **aggressor flow:** volume, dollar amount and trade count of aggressive buys vs
  aggressive sells, the largest trade, VWAP, and how many trades were missed (detected
  from gaps in Coinbase's trade IDs)
- **order book,** sampled every 5 seconds and averaged over the minute: spread, bid vs
  ask imbalance within 0.1%, 0.5% and 1% of the price, dollar depth within 1%, and the
  microprice

Coinbase's trade `side` field is the side of the **resting** order. The recorder flips
it, so "buy" means an aggressive buyer, which is the useful signal. This was checked
against the live order book: "BUY" prints happen at the bid.

The dashboard header shows how many days have been recorded and the coverage. Once
there's about a week of data, check whether any of it predicts returns:

```sh
docker run --rm -v /DATA/AppData/neatbase/data:/data ghcr.io/arandomenderman/neatbase:latest \
    python -m app.flowcheck
```

It reports, for each flow and book value:
- **IC:** the rank correlation with the next 1 and 4 hours' returns
- **t-stat:** whether the IC is distinguishable from zero
- **stability:** whether the IC keeps its sign in the first and second half of the data
- **top-vs-bottom-20% return spread,** compared with the round-trip fee

A value that is consistent in both halves with |t| > 3 is worth adding to the
network's inputs. A first look needs about a week of data, and a real answer about a
month.

## Development

```sh
pip install -r requirements.txt pytest
python -m pytest tests
DATA_DIR=./data GENERATIONS_PER_CYCLE=5 python -m app.trainer
DATA_DIR=./data python -m app.trader
```
