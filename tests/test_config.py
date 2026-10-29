import importlib
import json
import os

from app import config


def test_host_settings_file_and_key_file(tmp_path, monkeypatch):
    (tmp_path / "settings.env").write_text(
        '# comment\nPRODUCT_ID="ETH-USD"\nTAKER_FEE=0.006\nLIVE_MAX_USD = 25\n')
    (tmp_path / "cdp_api_key.json").write_text(json.dumps({
        "name": "organizations/o/apiKeys/k123456",
        "privateKey": "-----BEGIN EC PRIVATE KEY-----\nabc\n-----END EC PRIVATE KEY-----\n"}))
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("PRODUCT_ID", "SOL-USD")  # file wins over env
    try:
        c = importlib.reload(config)
        assert c.PRODUCT_ID == "ETH-USD" and c.TAKER_FEE == 0.006 and c.LIVE_MAX_USD == 25
        assert c.has_api_keys() and c.API_KEY_SOURCE == "cdp_api_key.json"
        assert c.COINBASE_API_SECRET.startswith("-----BEGIN EC PRIVATE KEY-----\nabc")
    finally:
        monkeypatch.undo()
        importlib.reload(config)
    assert not os.environ.get("COINBASE_API_KEY")
