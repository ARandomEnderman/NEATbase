"""Web UI access control: local networks only, a password chosen on first visit,
login rate limiting and CSRF tokens."""
import ipaddress
import json
import logging
import os
import secrets
import threading
import time
from datetime import timedelta

from flask import abort, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from . import config

log = logging.getLogger("web")

NETWORKS = [ipaddress.ip_network(n, strict=False) for n in config.ALLOWED_NETWORKS]
PUBLIC_ENDPOINTS = {"setup", "login", "healthz", "static"}
MAX_FAILURES, LOCKOUT_SECONDS = 5, 300
MIN_PASSWORD = 8

_failures = {}  # ip -> (count, locked_until)
_lock = threading.Lock()


def ip_allowed(addr):
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in n for n in NETWORKS)


def _load_auth():
    try:
        with open(config.AUTH_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def _save_auth(password, version):
    tmp = config.AUTH_PATH + ".tmp"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump({"hash": generate_password_hash(password), "version": version}, f)
    os.replace(tmp, config.AUTH_PATH)


def _secret_key():
    try:
        with open(config.SECRET_KEY_PATH, "rb") as f:
            return f.read()
    except FileNotFoundError:
        key = secrets.token_bytes(32)
        with open(os.open(config.SECRET_KEY_PATH, os.O_WRONLY | os.O_CREAT, 0o600), "wb") as f:
            f.write(key)
        return key


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def _check_password_rules(pw, confirm):
    if len(pw) < MIN_PASSWORD:
        return f"Password must be at least {MIN_PASSWORD} characters."
    if pw != confirm:
        return "Passwords don't match."
    return None


def _login(auth):
    session.clear()
    session.permanent = True
    session["user"] = True
    session["v"] = auth["version"]
    csrf_token()


def install(app):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    app.secret_key = _secret_key()
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_NAME="neatbase_session",
                      PERMANENT_SESSION_LIFETIME=timedelta(days=30))
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def guard():
        # request.remote_addr is the TCP peer; proxy headers are deliberately ignored.
        if not ip_allowed(request.remote_addr or ""):
            log.warning("Blocked web request from %s", request.remote_addr)
            abort(403)
        if request.method == "POST":
            sent = request.form.get("csrf") or request.headers.get("X-CSRF-Token")
            if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
                abort(400, "Invalid or missing CSRF token; reload the page.")
        if request.endpoint in PUBLIC_ENDPOINTS:
            return None
        auth = _load_auth()
        if auth is None:
            return redirect(url_for("setup"))
        if not session.get("user") or session.get("v") != auth["version"]:
            if request.path.startswith("/api/"):
                abort(401)
            return redirect(url_for("login"))
        return None

    @app.route("/setup", methods=["GET", "POST"])
    def setup():
        if _load_auth() is not None:
            return redirect(url_for("login"))
        error = None
        if request.method == "POST":
            error = _check_password_rules(request.form.get("password", ""),
                                          request.form.get("confirm", ""))
            if not error:
                _save_auth(request.form["password"], 1)
                log.info("Web UI password created from %s", request.remote_addr)
                _login(_load_auth())
                return redirect(url_for("index"))
        return render_template("auth.html", mode="setup", error=error)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        auth = _load_auth()
        if auth is None:
            return redirect(url_for("setup"))
        ip, error = request.remote_addr, None
        if request.method == "POST":
            with _lock:
                count, until = _failures.get(ip, (0, 0))
            if until > time.time():
                error = f"Too many attempts. Try again in {int(until - time.time()) + 1}s."
            elif check_password_hash(auth["hash"], request.form.get("password", "")):
                with _lock:
                    _failures.pop(ip, None)
                _login(auth)
                return redirect(url_for("index"))
            else:
                count += 1
                with _lock:
                    _failures[ip] = (0, time.time() + LOCKOUT_SECONDS) if count >= MAX_FAILURES \
                        else (count, 0)
                log.warning("Failed web login from %s", ip)
                error = "Wrong password."
        return render_template("auth.html", mode="login", error=error)

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/account/password", methods=["POST"])
    def change_password():
        auth = _load_auth()
        if not check_password_hash(auth["hash"], request.form.get("current", "")):
            return render_template("auth.html", mode="change", error="Current password is wrong.")
        error = _check_password_rules(request.form.get("password", ""), request.form.get("confirm", ""))
        if error:
            return render_template("auth.html", mode="change", error=error)
        _save_auth(request.form["password"], auth["version"] + 1)  # signs out other sessions
        _login(_load_auth())
        return redirect(url_for("index"))

    @app.route("/account")
    def account():
        return render_template("auth.html", mode="change", error=None)

    @app.route("/healthz")
    def healthz():
        return "ok"
