"""Single-admin authentication for the portal.

Password is stored as a pbkdf2-sha256 hash in settings (never in plain text).
Login mints an in-process session token (bearer or cookie) with a TTL — fine for
a single-operator tool; no external session store needed.
"""
import hashlib
import hmac
import os
import secrets
import time

from fastapi import Cookie, Header, HTTPException

from . import db
from .settings import K_ADMIN

_ITERATIONS = 240_000
_SESSION_TTL = int(os.getenv("ADMIN_SESSION_TTL", str(12 * 3600)))  # seconds
_sessions: dict[str, float] = {}   # token -> expiry (epoch seconds)


# ---------------------------------------------------------------- password
def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


def set_password(pw: str):
    db.set_setting(K_ADMIN, hash_password(pw))


def hash_api_key(key: str) -> str:
    """Fast hash for high-entropy API tokens (pbkdf2 is overkill for random keys)."""
    return hashlib.sha256(key.encode()).hexdigest()


def is_configured() -> bool:
    return bool(db.get_setting(K_ADMIN))


def bootstrap():
    """Seed the admin password from ADMIN_PASSWORD on first run (if not set yet)."""
    if not is_configured():
        env_pw = os.getenv("ADMIN_PASSWORD", "").strip()
        if env_pw:
            set_password(env_pw)


# ---------------------------------------------------------------- sessions
def login(pw: str) -> str | None:
    stored = db.get_setting(K_ADMIN)
    if not stored or not verify_password(pw, stored):
        return None
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + _SESSION_TTL
    return token


def logout(token: str):
    _sessions.pop(token, None)


def _valid(token: str | None) -> bool:
    if not token:
        return False
    exp = _sessions.get(token)
    if not exp:
        return False
    if exp < time.time():
        _sessions.pop(token, None)
        return False
    return True


def require_admin(authorization: str | None = Header(None),
                  admin_session: str | None = Cookie(None)) -> str:
    """FastAPI dependency: accept a Bearer token or the admin_session cookie."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    token = token or admin_session
    if not _valid(token):
        raise HTTPException(401, "admin authentication required")
    return token
