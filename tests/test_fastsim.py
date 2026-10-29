"""The compiled backtest must match sim.backtest exactly (it's only a speed-up)."""
import math
import random

import neat
import numpy as np
import pytest

from app import features, trainer
from app.fastsim import compile_ensemble, compile_net, ensemble_activate, fast_backtest
from app.sim import Costs, backtest

KEYS = ["fitness", "final_equity", "return_pct", "buy_hold_pct", "alpha_pct", "avg_exposure",
        "max_drawdown_pct", "trades", "win_rate", "fees_paid"]


def _market(seed, n=900):
    rnd, p, out = random.Random(seed), 100.0, []
    for i in range(n):
        o, p = p, p * math.exp(rnd.gauss(0.0002, 0.012))
        out.append((i * 3600, o, max(o, p) * (1 + rnd.random() * 0.006), min(o, p) * (1 - rnd.random() * 0.006),
                    p, rnd.uniform(1, 10)))
    return out


def _genomes(n=25, mutations=40):
    cfg = trainer.load_neat_config()
    out = list(neat.Population(cfg).population.values())[:n]
    rnd = random.Random(3)
    for g in out:
        for _ in range(rnd.randint(0, mutations)):  # grow hidden nodes and connections
            g.mutate(cfg.genome_config)
    return cfg, out


@pytest.mark.parametrize("order_type", ["maker", "taker"])
def test_compiled_backtest_matches_reference(order_type):
    cfg, genomes = _genomes()
    costs = Costs(order_type=order_type, spread_bps=3, slippage_bps=5)
    hidden = 0
    for seed in range(3):
        candles = _market(seed)
        feats = features.market_features(candles, 3600)
        arr = np.array(candles, dtype=np.float64)
        for g in genomes:
            hidden += len(g.nodes) > 1
            net = neat.nn.FeedForwardNetwork.create(g, cfg)
            for chunks in (1, 4):
                ref = backtest(net.activate, feats, candles, features.WARMUP, len(candles), costs, chunks=chunks)
                got = fast_backtest(compile_net(g, cfg), feats, arr, features.WARMUP, len(candles), costs,
                                    chunks=chunks)
                for k in KEYS:
                    assert math.isclose(got[k], ref[k], rel_tol=1e-9, abs_tol=1e-9), (k, got[k], ref[k])
    assert hidden > 10  # the comparison covered networks with hidden nodes


def test_compiled_network_matches_neat_activate():
    cfg, genomes = _genomes(10)
    rnd = random.Random(1)
    from app.fastsim import _activate
    for g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, cfg)
        node, bias, resp, lo, hi, src, w, outs, n_values = compile_net(g, cfg)
        for _ in range(20):
            x = [rnd.uniform(-1, 1) for _ in range(features.N_INPUTS)]
            assert math.isclose(_activate(np.array(x), np.zeros(n_values), node, bias, resp, lo, hi, src, w, outs),
                                net.activate(x)[0], rel_tol=1e-12, abs_tol=1e-12)


def test_compiled_ensemble_matches_reference():
    cfg, genomes = _genomes(12)
    costs = Costs(order_type="maker", spread_bps=3, slippage_bps=5)
    candles = _market(7)
    feats = features.market_features(candles, 3600)
    arr = np.array(candles, dtype=np.float64)
    for group in (genomes[:3], genomes[3:8], genomes[8:]):
        ref = backtest(ensemble_activate(group, cfg), feats, candles, features.WARMUP, len(candles), costs, chunks=4)
        got = fast_backtest(compile_ensemble(group, cfg), feats, arr, features.WARMUP, len(candles), costs, chunks=4)
        for k in KEYS:
            assert math.isclose(got[k], ref[k], rel_tol=1e-9, abs_tol=1e-9), (k, got[k], ref[k])
