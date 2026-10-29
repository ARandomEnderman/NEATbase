import importlib
import json
import os

from app import config


def test_host_settings_file_and_key_file(tmp_path, monkeypatch):
    (tmp_path / "settings.env").write_text(
        '# comment\nPRODUCT_IDS="ETH-USD, sol-usd"\nTAKER_FEE=0.006\nLIVE_MAX_USD = 25\n')
    (tmp_path / "cdp_api_key.json").write_text(json.dumps({
        "name": "organizations/o/apiKeys/k123456",
        "privateKey": "-----BEGIN EC PRIVATE KEY-----\nabc\n-----END EC PRIVATE KEY-----\n"}))
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("PRODUCT_IDS", "BTC-USD")  # file wins over env
    try:
        c = importlib.reload(config)
        assert c.PRODUCT_IDS == ["ETH-USD", "SOL-USD"] and c.TAKER_FEE == 0.006 and c.LIVE_MAX_USD == 25
        assert c.has_api_keys() and c.API_KEY_SOURCE == "cdp_api_key.json"
        assert c.COINBASE_API_SECRET.startswith("-----BEGIN EC PRIVATE KEY-----\nabc")
    finally:
        monkeypatch.undo()
        importlib.reload(config)
    assert not os.environ.get("COINBASE_API_KEY")
