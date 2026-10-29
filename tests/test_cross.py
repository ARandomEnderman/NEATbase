"""FEATURES=cross is a module-level switch, so check it in a fresh interpreter."""
import os
import subprocess
import sys

SCRIPT = r'''
import math, random
import numpy as np
from app import features, trainer
assert features.N_MARKET == 22 and features.N_INPUTS == 26
cfg = trainer.load_neat_config()
assert cfg.genome_config.num_inputs == 26 and len(cfg.genome_config.input_keys) == 26

rnd, n = random.Random(0), 700
def coin(drift):
    p, out = 100.0, []
    for i in range(n):
        if i == 400:
            continue  # a missing bar must not break alignment
        o, p = p, p * math.exp(drift + rnd.gauss(0, 0.01))
        out.append((i * 3600, o, max(o, p) * 1.001, min(o, p) * 0.999, p, 5.0))
    return out
market = {"A": coin(0.002), "B": coin(-0.002)}
basket = features.basket_index(market)
fa = features.market_features(market["A"], 3600, basket)
fb = features.market_features(market["B"], 3600, basket)
w = features.WARMUP
assert fa.shape == (len(market["A"]), 22) and np.isfinite(fa).all()
assert abs(fa[w:]).max() <= 1 + 1e-9
# A steadily outperforms the basket, B underperforms it
assert fa[w:, 18].mean() > 0.2 and fb[w:, 18].mean() < -0.2
# basket-trend inputs are identical for both coins (same basket)
assert np.allclose(fa[w + 10:w + 50, 16], fb[w + 10:w + 50, 16])
# day of week: a candle closing at 00:00 UTC on a Monday gives sin(0) = 0
import datetime
monday = int(datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc).timestamp()) - 3600
assert datetime.datetime.fromtimestamp(monday + 3600, datetime.timezone.utc).weekday() == 0
assert abs(features.market_features([(monday + i * 3600, 1, 1, 1, 1 + i * 1e-3, 1) for i in range(400)],
                                    3600, {monday + i * 3600: 0.0 for i in range(400)})[0, 20]) < 1e-9
print("ok")
'''


def test_cross_features_in_fresh_interpreter(tmp_path):
    env = {**os.environ, "FEATURES": "cross", "DATA_DIR": str(tmp_path), "CONFIG_DIR": str(tmp_path / "none")}
    r = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
