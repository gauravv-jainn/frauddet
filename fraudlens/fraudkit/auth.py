"""Users, roles and API keys.

* Passwords: PBKDF2-HMAC-SHA256, 600,000 iterations, 16-byte random salt (OWASP 2023 guidance).
* Lockout after ``max_failed_logins`` wrong passwords for ``lockout_minutes``.
* API keys: random 32-byte tokens shown once; only their SHA-256 hash is stored.
* Roles: viewer < analyst < admin. Every check goes through ``can(role, permission)``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from .db import Database, now

ROLES = ("viewer", "analyst", "admin")
PERMISSIONS: dict[str, set[str]] = {
    "view": {"viewer", "analyst", "admin"},
    "score": {"analyst", "admin"},  # batch scoring, manual scoring, what-if
    "decide_cases": {"analyst", "admin"},  # confirm / genuine / escalate / assign
    "feed_queue": {"analyst", "admin"},  # demo stream, add top-N, send batch to queue
    "clear_queue": {"admin"},
    "retrain": {"admin"},
    "manage_users": {"admin"},
    "manage_api_keys": {"admin"},
    "view_audit": {"admin"},
}
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LEN = 10
API_KEY_PREFIX = "fl_"


class AuthError(Exception):
    """Login or user-management failure with a message that is safe to show."""


def can(role: str | None, permission: str) -> bool:
    return bool(role) and role in PERMISSIONS.get(permission, set())


# --------------------------------------------------------------------------- passwords
def password_problems(password: str, username: str = "") -> list[str]:
    problems = []
    if len(password) < MIN_PASSWORD_LEN:
        problems.append(f"at least {MIN_PASSWORD_LEN} characters")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        problems.append("at least one letter and one digit")
    if username and username.lower() in password.lower():
        problems.append("not to contain the username")
    return problems


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=4)
def _dummy_hash(iterations: int) -> str:
    """Spend the same time on unknown usernames as on real ones (no user enumeration by timing)."""
    return hash_password("timing-equaliser-not-a-password-0", iterations=iterations)


@dataclass(frozen=True)
class User:
    username: str
    role: str
    active: bool
    must_change_pw: bool
    created_at: str
    last_login: str | None


def _user(row) -> User:
    return User(
        row["username"],
        row["role"],
        bool(row["active"]),
        bool(row["must_change_pw"]),
        row["created_at"],
        row["last_login"],
    )


class UserStore:
    def __init__(
        self, db: Database, max_failed: int = 5, lockout_minutes: int = 15, iterations: int = PBKDF2_ITERATIONS
    ):
        self.db = db
        self.max_failed = max_failed
        self.lockout = timedelta(minutes=lockout_minutes)
        self.iterations = iterations

    # ---- queries
    def count(self) -> int:
        with self.db.tx() as c:
            return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def has_active_admin(self) -> bool:
        with self.db.tx() as c:
            return c.execute("SELECT 1 FROM users WHERE role='admin' AND active=1").fetchone() is not None

    def get(self, username: str) -> User | None:
        with self.db.tx() as c:
            row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return _user(row) if row else None

    def list(self) -> list[User]:
        with self.db.tx() as c:
            return [_user(r) for r in c.execute("SELECT * FROM users ORDER BY username")]

    # ---- changes
    def create(
        self, username: str, password: str, role: str, actor: str | None = None, must_change_pw: bool = False
    ) -> User:
        username = username.strip().lower()
        if not USERNAME_RE.match(username):
            raise AuthError("Username must be 3-32 characters: lowercase letters, digits, '.', '_' or '-'.")
        if role not in ROLES:
            raise AuthError(f"Role must be one of {', '.join(ROLES)}.")
        problems = password_problems(password, username)
        if problems:
            raise AuthError("Password needs " + "; ".join(problems) + ".")
        if self.get(username):
            raise AuthError("That username is already taken.")
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO users (username, password_hash, role, must_change_pw, created_at) VALUES (?,?,?,?,?)",
                (username, hash_password(password, iterations=self.iterations), role, int(must_change_pw), now()),
            )
        self.db.log_admin(actor or username, "user_created", username, f"role={role}")
        return self.get(username)

    def authenticate(self, username: str, password: str) -> User:
        """Return the user or raise AuthError with a generic message (no hint whether the name exists)."""
        username = (username or "").strip().lower()
        generic = AuthError("Invalid username or password.")
        with self.db.tx() as c:
            row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            verify_password(password, _dummy_hash(self.iterations))
            raise generic
        if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > datetime.now(UTC):
            raise AuthError("Account temporarily locked after too many failed attempts. Try again later.")
        if not verify_password(password, row["password_hash"]):
            failed = row["failed_attempts"] + 1
            locked = (
                (datetime.now(UTC) + self.lockout).isoformat(timespec="seconds") if failed >= self.max_failed else None
            )
            with self.db.tx() as c:
                c.execute(
                    "UPDATE users SET failed_attempts=?, locked_until=? WHERE username=?",
                    (0 if locked else failed, locked, username),
                )
            self.db.log_admin(username, "login_failed" + ("_locked" if locked else ""), username)
            raise generic
        if not row["active"]:
            raise AuthError("This account is disabled. Ask an administrator.")
        with self.db.tx() as c:
            c.execute(
                "UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE username=?",
                (now(), username),
            )
        self.db.log_admin(username, "login", username)
        return self.get(username)

    def set_role(self, username: str, role: str, actor: str):
        if role not in ROLES:
            raise AuthError("Unknown role.")
        user = self.get(username)
        if user is None:
            raise AuthError("Unknown user.")
        if user.role == "admin" and role != "admin" and self._last_admin(username):
            raise AuthError("Cannot demote the last active admin.")
        with self.db.tx() as c:
            c.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        self.db.log_admin(actor, "role_changed", username, f"{user.role} -> {role}")

    def set_active(self, username: str, active: bool, actor: str):
        user = self.get(username)
        if user is None:
            raise AuthError("Unknown user.")
        if not active and user.role == "admin" and self._last_admin(username):
            raise AuthError("Cannot disable the last active admin.")
        if not active and username == actor:
            raise AuthError("You cannot disable your own account.")
        with self.db.tx() as c:
            c.execute("UPDATE users SET active=? WHERE username=?", (int(active), username))
        self.db.log_admin(actor, "user_enabled" if active else "user_disabled", username)

    def set_password(self, username: str, password: str, actor: str, must_change: bool = False):
        problems = password_problems(password, username)
        if problems:
            raise AuthError("Password needs " + "; ".join(problems) + ".")
        with self.db.tx() as c:
            cur = c.execute(
                "UPDATE users SET password_hash=?, must_change_pw=?, failed_attempts=0, locked_until=NULL "
                "WHERE username=?",
                (hash_password(password, iterations=self.iterations), int(must_change), username),
            )
            if cur.rowcount == 0:
                raise AuthError("Unknown user.")
        self.db.log_admin(actor, "password_set" if actor == username else "password_reset", username)

    def _last_admin(self, username: str) -> bool:
        with self.db.tx() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1 AND username<>?", (username,)
            ).fetchone()[0]
        return n == 0


# --------------------------------------------------------------------------- API keys
@dataclass(frozen=True)
class ApiKey:
    key_id: str
    name: str
    created_by: str | None
    created_at: str
    last_used: str | None
    use_count: int
    revoked_at: str | None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


def _key_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class ApiKeyStore:
    def __init__(self, db: Database):
        self.db = db

    def create(self, name: str, actor: str) -> tuple[ApiKey, str]:
        """Returns the record and the plaintext key. The plaintext is never stored and cannot be shown again."""
        name = name.strip()
        if not 2 <= len(name) <= 64:
            raise AuthError("Key name must be 2-64 characters.")
        key_id = "key_" + secrets.token_hex(4)
        token = API_KEY_PREFIX + secrets.token_urlsafe(32)
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO api_keys (key_id, name, key_hash, created_by, created_at) VALUES (?,?,?,?,?)",
                (key_id, name, _key_hash(token), actor, now()),
            )
        self.db.log_admin(actor, "api_key_created", key_id, name)
        return self.get(key_id), token

    def verify(self, token: str | None) -> ApiKey | None:
        if not token or not token.startswith(API_KEY_PREFIX):
            return None
        with self.db.tx() as c:
            row = c.execute(
                "SELECT * FROM api_keys WHERE key_hash=? AND revoked_at IS NULL", (_key_hash(token),)
            ).fetchone()
            if row is None:
                return None
            c.execute("UPDATE api_keys SET last_used=?, use_count=use_count+1 WHERE key_id=?", (now(), row["key_id"]))
        return self.get(row["key_id"])

    def revoke(self, key_id: str, actor: str):
        with self.db.tx() as c:
            cur = c.execute("UPDATE api_keys SET revoked_at=? WHERE key_id=? AND revoked_at IS NULL", (now(), key_id))
            if cur.rowcount == 0:
                raise AuthError("Unknown or already revoked key.")
        self.db.log_admin(actor, "api_key_revoked", key_id)

    def get(self, key_id: str) -> ApiKey | None:
        with self.db.tx() as c:
            r = c.execute("SELECT * FROM api_keys WHERE key_id=?", (key_id,)).fetchone()
        return (
            ApiKey(
                r["key_id"],
                r["name"],
                r["created_by"],
                r["created_at"],
                r["last_used"],
                r["use_count"],
                r["revoked_at"],
            )
            if r
            else None
        )

    def list(self) -> list[ApiKey]:
        with self.db.tx() as c:
            ids = [r["key_id"] for r in c.execute("SELECT key_id FROM api_keys ORDER BY created_at DESC")]
        return [self.get(i) for i in ids]
