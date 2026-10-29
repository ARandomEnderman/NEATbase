import os
import re

import pytest

from app import auth, config
from app.web import app

LAN = {"REMOTE_ADDR": "192.168.1.20"}


@pytest.fixture
def client():
    for p in (config.AUTH_PATH, config.KILL_SWITCH_PATH):
        if os.path.exists(p):
            os.remove(p)
    auth._failures.clear()
    return app.test_client()


def _csrf(resp):
    return re.search(r'name="csrf" value="([^"]+)"', resp.get_data(as_text=True)).group(1)


@pytest.mark.parametrize("ip,ok", [
    ("192.168.1.5", True), ("10.1.2.3", True), ("172.20.0.1", True), ("127.0.0.1", True),
    ("100.101.102.103", True), ("fd7a:115c:a1e0::1", True), ("::ffff:192.168.0.9", True),
    ("8.8.8.8", False), ("100.128.0.1", False), ("2001:4860:4860::8888", False), ("garbage", False),
])
def test_ip_allowlist(ip, ok):
    assert auth.ip_allowed(ip) is ok


def test_public_ip_is_rejected_everywhere(client):
    for path in ("/", "/setup", "/login", "/api/summary", "/healthz"):
        assert client.get(path, environ_base={"REMOTE_ADDR": "8.8.8.8"}).status_code == 403


def test_first_visit_sets_password_then_login_required(client):
    assert client.get("/", environ_base=LAN).headers["Location"].endswith("/setup")
    page = client.get("/setup", environ_base=LAN)
    r = client.post("/setup", environ_base=LAN, data={
        "csrf": _csrf(page), "password": "hunter2hunter2", "confirm": "hunter2hunter2"})
    assert r.status_code == 302 and client.get("/", environ_base=LAN).status_code == 200
    assert client.get("/api/summary", environ_base=LAN).status_code == 200

    # setup can't be reused to overwrite the password
    assert client.get("/setup", environ_base=LAN).headers["Location"].endswith("/login")

    fresh = app.test_client()
    assert fresh.get("/api/summary", environ_base=LAN).status_code == 401
    assert fresh.get("/", environ_base=LAN).headers["Location"].endswith("/login")
    page = fresh.get("/login", environ_base=LAN)
    bad = fresh.post("/login", environ_base=LAN, data={"csrf": _csrf(page), "password": "nope"})
    assert "Wrong password" in bad.get_data(as_text=True)
    ok = fresh.post("/login", environ_base=LAN, data={"csrf": _csrf(page), "password": "hunter2hunter2"})
    assert ok.status_code == 302 and fresh.get("/", environ_base=LAN).status_code == 200


def test_post_without_csrf_is_rejected(client):
    assert client.post("/setup", environ_base=LAN,
                       data={"password": "x" * 10, "confirm": "x" * 10}).status_code == 400
    assert not os.path.exists(config.AUTH_PATH)


def test_lockout_after_repeated_failures(client):
    page = client.get("/setup", environ_base=LAN)
    client.post("/setup", environ_base=LAN, data={"csrf": _csrf(page), "password": "correcthorse",
                                                   "confirm": "correcthorse"})
    c = app.test_client()
    tok = _csrf(c.get("/login", environ_base=LAN))
    for _ in range(auth.MAX_FAILURES):
        c.post("/login", environ_base=LAN, data={"csrf": tok, "password": "wrong"})
    r = c.post("/login", environ_base=LAN, data={"csrf": tok, "password": "correcthorse"})
    assert "Too many attempts" in r.get_data(as_text=True)


def test_password_change_signs_out_other_sessions_and_pause_toggle(client):
    page = client.get("/setup", environ_base=LAN)
    client.post("/setup", environ_base=LAN, data={"csrf": _csrf(page), "password": "firstpass1",
                                                   "confirm": "firstpass1"})
    other = app.test_client()
    tok = _csrf(other.get("/login", environ_base=LAN))
    other.post("/login", environ_base=LAN, data={"csrf": tok, "password": "firstpass1"})
    assert other.get("/api/summary", environ_base=LAN).status_code == 200

    tok = _csrf(client.get("/account", environ_base=LAN))
    client.post("/account/password", environ_base=LAN, data={
        "csrf": tok, "current": "firstpass1", "password": "secondpass2", "confirm": "secondpass2"})
    assert other.get("/api/summary", environ_base=LAN).status_code == 401
    assert client.get("/api/summary", environ_base=LAN).status_code == 200

    tok = _csrf(client.get("/account", environ_base=LAN))
    r = client.post("/api/pause", environ_base=LAN, json={"paused": True}, headers={"X-CSRF-Token": tok})
    assert r.get_json() == {"paused": True} and os.path.exists(config.KILL_SWITCH_PATH)
    client.post("/api/pause", environ_base=LAN, json={"paused": False}, headers={"X-CSRF-Token": tok})
    assert not os.path.exists(config.KILL_SWITCH_PATH)
