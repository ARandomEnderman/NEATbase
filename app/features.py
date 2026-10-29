"""Network inputs. The exact same code feeds training and live trading."""
import math

import numpy as np

WARMUP = 300  # bars needed before indicators are meaningful (EMA200 has settled)
N_MARKET = 16
N_POSITION = 4
N_INPUTS = N_MARKET + N_POSITION


def _ema(x, span):
    a = 2.0 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _rolling(x, w, fn):
    out = np.full_like(x, np.nan)
    for i in range(w - 1, len(x)):
        out[i] = fn(x[i - w + 1:i + 1])
    return out


def _rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    a = 1.0 / n
    au, ad = np.empty_like(close), np.empty_like(close)
    au[0], ad[0] = up[0], dn[0]
    for i in range(1, len(close)):
        au[i] = a * up[i] + (1 - a) * au[i - 1]
        ad[i] = a * dn[i] + (1 - a) * ad[i - 1]
    rs = au / np.maximum(ad, 1e-12)
    return 100 - 100 / (1 + rs)


def market_features(candles, granularity):
    """candles: rows of (start, open, high, low, close, volume), oldest first.
    Returns an (n, N_MARKET) array; rows before WARMUP are not valid."""
    c = np.asarray(candles, dtype=float)
    t, o, h, l, cl, v = c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4], c[:, 5]
    n = len(cl)
    logc = np.log(cl)
    lr = np.diff(logc, prepend=logc[0])
    vol20 = np.maximum(_rolling(lr, 20, np.std), 1e-6)
    vol100 = np.maximum(_rolling(lr, 100, np.std), 1e-6)

    week = max(2, round(7 * 86400 / granularity))  # one week of bars, for longer-range context
    f = np.zeros((n, N_MARKET))
    for j, k in enumerate((1, 4, 16, 64)):
        past = np.concatenate([np.full(k, logc[0]), logc[:-k]])
        f[:, j] = np.clip((logc - past) / (vol20 * math.sqrt(k)), -3, 3) / 3
    f[:, 4] = (_rsi(cl) - 50) / 50
    ema20, ema50 = _ema(cl, 20), _ema(cl, 50)
    f[:, 5] = np.clip((logc - np.log(ema20)) / vol20, -3, 3) / 3
    f[:, 6] = np.clip((np.log(ema20) - np.log(ema50)) / vol20, -5, 5) / 5
    f[:, 7] = np.clip(np.log(vol20 / vol100), -2, 2) / 2
    lv = np.log1p(v)
    lv_mean = _rolling(lv, 50, np.mean)
    lv_std = np.maximum(_rolling(lv, 50, np.std), 1e-6)
    f[:, 8] = np.clip((lv - lv_mean) / lv_std, -3, 3) / 3
    hi20, lo20 = _rolling(h, 20, np.max), _rolling(l, 20, np.min)
    f[:, 9] = (cl - lo20) / np.maximum(hi20 - lo20, 1e-12) * 2 - 1
    f[:, 10] = (cl - o) / np.maximum(h - l, 1e-12)
    hour = ((t + granularity) % 86400) / 86400 * 2 * math.pi  # time at candle close
    f[:, 11] = np.sin(hour)
    f[:, 12] = np.cos(hour)
    past = np.concatenate([np.full(week, logc[0]), logc[:-week]])
    f[:, 13] = np.clip((logc - past) / (vol20 * math.sqrt(week)), -3, 3) / 3
    hi_w, lo_w = _rolling(h, min(week, WARMUP), np.max), _rolling(l, min(week, WARMUP), np.min)
    f[:, 14] = (cl - lo_w) / np.maximum(hi_w - lo_w, 1e-12) * 2 - 1
    f[:, 15] = np.clip((logc - np.log(_ema(cl, 200))) / (vol20 * 10), -3, 3) / 3
    return np.nan_to_num(f)


def position_features(in_position, unrealized_pnl, bars_held, prev_output):
    """prev_output is the network's own output on the previous bar: a one-step memory."""
    return [1.0 if in_position else -1.0,
            math.tanh(unrealized_pnl * 10),
            math.tanh(bars_held / 48),
            prev_output]
