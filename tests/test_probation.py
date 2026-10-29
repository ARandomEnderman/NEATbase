import json
import os

import neat
import pytest

from app import config, trainer
from app.trader import apply_fill

DAY = 86400


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHAMPION_PATH", str(tmp_path / "champion.pkl"))
    monkeypatch.setattr(config, "CHALLENGERS_DIR", str(tmp_path / "challengers"))
    monkeypatch.setattr(config, "TRAINER_DB", str(tmp_path / "trainer.db"))
    cfg = trainer.load_neat_config()
    genomes = list(neat.Population(cfg).population.values())[:4]
    scores = {}  # genome key -> forward fitness the fake backtest reports

    def fake_evaluate(genome, cfg_, windows, costs, **kw):
        return {"fitness": scores[genome.key], "return_pct": scores[genome.key], "trades": 3}

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    monkeypatch.setattr(trainer, "windows", lambda market, a, b: {})
    return trainer.open_db(), cfg, genomes, scores


def _status(db):
    return dict(db.execute("SELECT id, status FROM challengers").fetchall())


def test_challenger_must_wait_then_win_on_forward_data(env):
    db, cfg, (champ, ch_win, ch_lose, ch_young), scores = env
    now = 100 * DAY
    trainer.save_champion(champ, {"generation": 1})
    scores.update({champ.key: 1.0, ch_win.key: 2.0, ch_lose.key: 0.5, ch_young.key: 9.0})
    trainer.add_challenger(db, ch_win, now - 8 * DAY, 2, {"fitness": 1})
    trainer.add_challenger(db, ch_lose, now - 10 * DAY, 3, {"fitness": 1})
    trainer.add_challenger(db, ch_young, now - 2 * DAY, 4, {"fitness": 1})  # still on probation
    db.commit()

    trainer.run_probations(db, cfg, {}, None, now)
    st = sorted(_status(db).values())
    assert st == ["lost", "promoted", "waiting"]
    assert trainer.load_champion()["genome"].key == ch_win.key
    assert len(os.listdir(config.CHALLENGERS_DIR)) == 1  # only the young one is kept

    trainer.run_probations(db, cfg, {}, None, now)  # decided challengers are never retested
    assert sorted(_status(db).values()) == ["lost", "promoted", "waiting"]


def test_one_challenger_per_day_keeps_the_best(env):
    db, cfg, (champ, a, b, c), scores = env
    now = 100 * DAY
    assert trainer.add_challenger(db, a, now, 2, {"fitness": 1.0}).startswith("challenger")
    assert "better" in trainer.add_challenger(db, b, now + 3600, 3, {"fitness": 0.5})
    assert trainer.add_challenger(db, c, now + 7200, 4, {"fitness": 2.0}).startswith("challenger")
    rows = db.execute("SELECT status, validation FROM challengers ORDER BY born").fetchall()
    assert [(s, json.loads(v)["fitness"]) for s, v in rows] == [("replaced", 1.0), ("waiting", 2.0)]


def test_apply_fill_handles_partial_sells():
    slot = {"cash": 0.0, "qty": 0.0, "entry_cost": 0.0, "held": 5}
    apply_fill(slot, {"side": "buy", "qty": 2.0, "cash_delta": -200.0}, 100.0)
    assert slot == {"cash": -200.0, "qty": 2.0, "entry_cost": 200.0, "held": 0}
    pnl = apply_fill(slot, {"side": "sell", "qty": 0.5, "cash_delta": 60.0}, 120.0)
    assert pnl == 10.0 and slot["qty"] == 1.5 and slot["entry_cost"] == 150.0
    pnl = apply_fill(slot, {"side": "sell", "qty": 1.5, "cash_delta": 150.0}, 100.0)
    assert pnl == 0.0 and slot["qty"] == 0.0 and slot["entry_cost"] == 0.0
