"""Numba-compiled copy of sim.backtest for training speed (~50x).

sim.py stays the reference implementation; tests/test_fastsim.py checks that both give
the same results on random networks and markets. The trader never uses this file.
Recorded runs (equity curves, trade logs) go through sim.py.
"""
import math

import neat
import numpy as np
from numba import njit

from . import config
from .features import N_MARKET
from .sim import backtest, backtest_many

R_FITNESS, R_FINAL, R_RETURN, R_BH, R_ALPHA, R_EXPOSURE, R_DD, R_TRADES, R_WINS, R_FEES = range(10)


def compile_net(genome, cfg):
    """Flatten neat-python's own evaluation order into arrays."""
    net = neat.nn.FeedForwardNetwork.create(genome, cfg)
    index = {k: i for i, k in enumerate(net.input_nodes)}
    for node, *_ in net.node_evals:
        index.setdefault(node, len(index))
    for k in net.output_nodes:
        index.setdefault(k, len(index))  # an output with no inputs stays 0, like neat-python
    for _, act, agg, *_ in net.node_evals:
        if act is not neat.activations.tanh_activation or agg is not neat.aggregations.sum_aggregation:
            raise ValueError("fastsim only supports tanh activation with sum aggregation")
    node, bias, resp, lo, hi, src, w = [], [], [], [], [], [], []
    for n, _, _, b, r, links in net.node_evals:
        node.append(index[n])
        bias.append(b)
        resp.append(r)
        lo.append(len(src))
        for i, wt in links:
            src.append(index[i])
            w.append(wt)
        hi.append(len(src))
    return (np.array(node, np.int64), np.array(bias, np.float64), np.array(resp, np.float64),
            np.array(lo, np.int64), np.array(hi, np.int64), np.array(src, np.int64),
            np.array(w, np.float64), np.array([index[net.output_nodes[0]]], np.int64), len(index))


def compile_ensemble(genomes, cfg):
    """Several networks sharing the same inputs; the output is the mean of theirs."""
    n_in = len(cfg.genome_config.input_keys)
    parts = [compile_net(g, cfg) for g in genomes]
    node, bias, resp, lo, hi, src, w, outs = [], [], [], [], [], [], [], []
    n_values, n_links = n_in, 0

    def shift(a, off):
        return np.where(a >= n_in, a + off, a)

    for p_node, p_bias, p_resp, p_lo, p_hi, p_src, p_w, p_out, p_n in parts:
        off = n_values - n_in
        node.append(shift(p_node, off))
        bias.append(p_bias)
        resp.append(p_resp)
        lo.append(p_lo + n_links)
        hi.append(p_hi + n_links)
        src.append(shift(p_src, off))
        w.append(p_w)
        outs.append(shift(p_out, off))
        n_values += p_n - n_in
        n_links += len(p_src)
    cat = np.concatenate
    return (cat(node), cat(bias), cat(resp), cat(lo), cat(hi), cat(src), cat(w), cat(outs), n_values)


@njit(cache=True)
def _activate(x, values, node, bias, resp, lo, hi, src, w, outs):
    for k in range(x.shape[0]):
        values[k] = x[k]
    for e in range(node.shape[0]):
        s = 0.0
        for j in range(lo[e], hi[e]):
            s += values[src[j]] * w[j]
        z = 2.5 * (bias[e] + resp[e] * s)
        z = max(-60.0, min(60.0, z))
        values[node[e]] = math.tanh(z)
    o = 0.0
    for k in range(outs.shape[0]):
        o += values[outs[k]]
    return o / outs.shape[0]


@njit(cache=True)
def _net_sale(qty, price, fee_rate):
    notional = qty * price
    return notional - notional * fee_rate  # same float ops as sim.sell_base


@njit(cache=True)
def _backtest(node, bias, resp, lo, hi, src, w, outs, n_values, feats, candles, start, end,
              taker, maker_fee, slip, spread, maker, fallback, start_cash, max_fraction, chunks,
              buy_th, sell_th, steps, dd_pen, inact_pen, alpha_mode):
    values = np.zeros(n_values)
    x = np.zeros(feats.shape[1] + 4)
    half_spread = spread / 20_000
    slip_f = slip / 10_000
    cash, qty, entry_cost, held, prev_out = start_cash, 0.0, 0.0, 0, 0.0
    p_side, p_price, p_amount = 0, 0.0, 0.0  # pending maker order; side 0 = none, 1 = buy, 2 = sell
    sell_fails, trades, wins, fees = 0, 0, 0, 0.0
    peak, max_dd = start_cash, 0.0
    bounds = np.empty(chunks, np.int64)
    for k in range(1, chunks + 1):
        bounds[k - 1] = start + (end - start) * k // chunks
    chunk_eq = np.empty(chunks + 1)
    chunk_eq[0] = start_cash
    n_chunks = 0
    alphas = np.zeros(chunks)
    n_alpha = 0
    chunk_px, chunk_exp, chunk_bars = candles[start, 4], 0.0, 0
    prev_exp, exp_sum = 0.0, 0.0

    for i in range(start, end):
        lo_, hi_, close = candles[i, 3], candles[i, 2], candles[i, 4]
        if p_side != 0:
            side, price, amount = p_side, p_price, p_amount
            p_side = 0
            filled = lo_ < price if side == 1 else hi_ > price
            if filled:
                if side == 1:
                    notional = amount / (1 + maker_fee)
                    fee = notional * maker_fee
                    cash -= amount
                    qty, entry_cost, held, fees = notional / price, amount, 0, fees + fee
                else:
                    notional = qty * price
                    fee = notional * maker_fee
                    got = notional - fee
                    cash += got
                    trades += 1
                    if got > entry_cost:
                        wins += 1
                    fees += fee
                    qty, sell_fails = 0.0, 0
            elif side == 2:
                sell_fails += 1

        sell_px = close * (1 - half_spread - slip_f)
        in_pos = qty > 0
        value = _net_sale(qty, sell_px, taker) if in_pos else 0.0
        pnl = value / entry_cost - 1 if in_pos else 0.0
        total = cash + value
        expo = value / total / max_fraction if total > 0 else 0.0
        for k in range(feats.shape[1]):
            x[k] = feats[i, k]
        nf = feats.shape[1]
        x[nf] = 2 * min(1.0, expo) - 1
        x[nf + 1] = math.tanh(pnl * 10)
        x[nf + 2] = math.tanh(held / 48)
        x[nf + 3] = prev_out
        o = _activate(x, values, node, bias, resp, lo, hi, src, w, outs)
        prev_out = o
        action = 0  # 0 none, 1 buy, 2 sell
        if not in_pos and o > buy_th:
            action = 1
        elif in_pos and o < sell_th:
            action = 2
        if action != 2:
            sell_fails = 0

        spend = 0.0
        if action == 1:
            strength = (o - buy_th) / (1 - buy_th)
            k = min(steps, max(1, math.ceil(strength * steps)))
            spend = cash * (max_fraction * k / steps)
        if action == 1 and spend >= 1.0:
            if maker:
                p_side, p_price, p_amount = 1, close * (1 - half_spread), spend
            else:
                price = close * (1 + half_spread + slip_f)
                notional = spend / (1 + taker)
                fee = notional * taker
                cash -= spend
                qty, entry_cost, held, fees = notional / price, spend, 0, fees + fee
        elif action == 2:
            if maker and sell_fails < fallback:
                p_side, p_price, p_amount = 2, close * (1 + half_spread), qty
            else:
                notional = qty * sell_px
                fee = notional * taker
                got = notional - fee
                cash += got
                trades += 1
                if got > entry_cost:
                    wins += 1
                fees += fee
                qty, sell_fails = 0.0, 0
        elif qty > 0:
            held += 1

        value = _net_sale(qty, sell_px, taker) if qty > 0 else 0.0
        equity = cash + value
        chunk_exp += prev_exp
        chunk_bars += 1
        exp_sum += prev_exp
        prev_exp = value / equity if equity > 0 else 0.0
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
        for k in range(chunks):
            if bounds[k] == i + 1:
                avg_exp = chunk_exp / max(1, chunk_bars)
                alphas[n_alpha] = 100 * (math.log(max(equity, 1e-9) / chunk_eq[n_chunks])
                                         - avg_exp * math.log(close / chunk_px))
                n_alpha += 1
                n_chunks += 1
                chunk_eq[n_chunks] = equity
                chunk_px, chunk_exp, chunk_bars = close, 0.0, 0
                break

    if qty > 0:
        trades += 1

    n_rets = n_alpha if alpha_mode else n_chunks
    rets = np.empty(n_rets)
    for k in range(n_rets):
        if alpha_mode:
            rets[k] = alphas[k]
        else:
            rets[k] = 100 * math.log(max(chunk_eq[k + 1], 1e-9) / chunk_eq[k])
    total_r = 0.0
    for k in range(n_rets):
        total_r += rets[k]
    std = 0.0
    if n_rets > 1:
        m = total_r / n_rets
        acc = 0.0
        for k in range(n_rets):
            acc += (rets[k] - m) ** 2
        std = math.sqrt(acc / n_rets)
    fitness = total_r - std - dd_pen * 100 * max_dd
    if trades == 0:
        fitness -= inact_pen

    first, last = candles[start, 4], candles[end - 1, 4]
    bh_qty = start_cash / (1 + taker) / (first * (1 + half_spread + slip_f))
    bh = _net_sale(bh_qty, last * (1 - half_spread - slip_f), taker)
    alpha_total = 0.0
    for k in range(n_alpha):
        alpha_total += alphas[k]
    final = chunk_eq[n_chunks]
    res = np.empty(10)
    res[0] = fitness
    res[1] = final
    res[2] = 100 * (final / start_cash - 1)
    res[3] = 100 * (bh / start_cash - 1)
    res[4] = alpha_total
    res[5] = exp_sum / max(1, end - start)
    res[6] = 100 * max_dd
    res[7] = trades
    res[8] = wins
    res[9] = fees
    return res


def fast_backtest(compiled, feats, candles, start, end, costs, start_cash=1000.0,
                  max_fraction=None, chunks=1):
    max_fraction = config.MAX_POSITION_FRACTION if max_fraction is None else max_fraction
    r = _backtest(*compiled, np.ascontiguousarray(feats, np.float64), np.ascontiguousarray(candles, np.float64),
                  start, end, costs.taker_fee, costs.maker_fee, costs.slippage_bps, costs.spread_bps,
                  costs.order_type == "maker", costs.fallback_bars, start_cash, max_fraction, chunks,
                  config.BUY_THRESHOLD, config.SELL_THRESHOLD, config.SIZE_STEPS, config.DRAWDOWN_PENALTY,
                  config.INACTIVITY_PENALTY, config.FITNESS_MODE == "alpha")
    trades = int(r[R_TRADES])
    return {"fitness": r[R_FITNESS], "final_equity": r[R_FINAL], "return_pct": r[R_RETURN],
            "buy_hold_pct": r[R_BH], "alpha_pct": r[R_ALPHA], "avg_exposure": r[R_EXPOSURE],
            "max_drawdown_pct": r[R_DD], "trades": trades,
            "win_rate": r[R_WINS] / trades if trades else 0.0, "fees_paid": r[R_FEES]}


def ensemble_activate(genomes, cfg):
    """Python reference for an ensemble: the mean of the networks' outputs."""
    nets = [neat.nn.FeedForwardNetwork.create(g, cfg) for g in genomes]
    return lambda x: [sum(n.activate(x)[0] for n in nets) / len(nets)]


def evaluate(genome, cfg, windows, costs, chunks=1, record=False):
    """backtest_many for a genome (or a list of genomes = an ensemble): compiled, or
    through sim.py when recording curves."""
    genomes = genome if isinstance(genome, (list, tuple)) else [genome]
    if record:
        return backtest_many(ensemble_activate(genomes, cfg), windows, costs, chunks=chunks, record=True)
    compiled = compile_ensemble(genomes, cfg) if len(genomes) > 1 else compile_net(genomes[0], cfg)
    per = {pid: fast_backtest(compiled, f, c, a, b, costs, chunks=chunks)
           for pid, (f, c, a, b) in windows.items() if b - a >= 24}
    if not per:
        raise RuntimeError("No coin has enough candles in this window")
    rs = list(per.values())
    n = len(rs)
    return {
        "fitness": sum(r["fitness"] for r in rs) / n,
        "return_pct": sum(r["return_pct"] for r in rs) / n,
        "buy_hold_pct": sum(r["buy_hold_pct"] for r in rs) / n,
        "alpha_pct": sum(r["alpha_pct"] for r in rs) / n,
        "avg_exposure": sum(r["avg_exposure"] for r in rs) / n,
        "max_drawdown_pct": max(r["max_drawdown_pct"] for r in rs),
        "trades": sum(r["trades"] for r in rs),
        "fees_paid": sum(r["fees_paid"] for r in rs),
        "per_product": per,
    }


__all__ = ["compile_net", "compile_ensemble", "fast_backtest", "evaluate", "backtest", "N_MARKET"]
