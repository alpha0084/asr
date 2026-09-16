"""Multi-admin authentication for the portal.

Admins are rows in the `admin_users` table (email + pbkdf2-sha256 password hash).
Login mints an in-process session token (bearer or cookie) with a TTL that carries
the admin's identity. The first admin is seeded on boot from ADMIN_EMAIL +
ADMIN_PASSWORD; after that, admins are created from the portal.
"""
import hashlib
import hmac
import os
import secrets
import time
import uuid
from datetime import datetime, timezone

from fastapi import Cookie, Header, HTTPException

from . import config, db

_ITERATIONS = 240_000
_SESSION_TTL = int(os.getenv("ADMIN_SESSION_TTL", str(12 * 3600)))  # seconds
# token -> {"exp": epoch, "id":.., "email":.., "name":..}
_sessions: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- password hashing
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


def hash_api_key(key: str) -> str:
    """Fast hash for high-entropy API tokens (pbkdf2 is overkill for random keys)."""
    return hashlib.sha256(key.encode()).hexdigest()


# ---------------------------------------------------------------- admin users
def create_admin(email: str, password: str, name: str = "") -> dict:
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("a valid email is required")
    if len(password or "") < 6:
        raise ValueError("password must be at least 6 characters")
    if db.get_admin_by_email(email):
        raise ValueError("an admin with that email already exists")
    uid = uuid.uuid4().hex[:12]
    db.add_admin_user(uid, email, name.strip() or email.split("@")[0], hash_password(password), _now())
    return {"id": uid, "email": email, "name": name.strip() or email.split("@")[0]}


def is_configured() -> bool:
    return db.count_admin_users() > 0


def bootstrap():
    """Seed the first admin from ADMIN_EMAIL + ADMIN_PASSWORD on first run."""
    if is_configured():
        return
    pw = os.getenv("ADMIN_PASSWORD", "").strip()
    if pw:
        try:
            create_admin(config.ADMIN_EMAIL, pw, "Administrator")
        except ValueError:
            pass


# ---------------------------------------------------------------- sessions
def login(email: str, password: str) -> dict | None:
    user = db.get_admin_by_email((email or "").strip().lower())
    if not user or not verify_password(password, user.get("password_hash", "")):
        return None
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"exp": time.time() + _SESSION_TTL, "id": user["id"],
                        "email": user["email"], "name": user.get("name") or user["email"]}
    db.set_admin_last_login(user["id"], _now())
    return {"token": token, "email": user["email"], "name": user.get("name") or user["email"]}


def logout(token: str):
    _sessions.pop(token, None)


def _session(token: str | None) -> dict | None:
    if not token:
        return None
    s = _sessions.get(token)
    if not s:
        return None
    if s["exp"] < time.time():
        _sessions.pop(token, None)
        return None
    return s


def require_admin(authorization: str | None = Header(None),
                  admin_session: str | None = Cookie(None)) -> dict:
    """FastAPI dependency: accept a Bearer token or the admin_session cookie.

    Returns the session identity dict {token, id, email, name}.
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    token = token or admin_session
    s = _session(token)
    if not s:
        raise HTTPException(401, "admin authentication required")
    return {"token": token, **s}
