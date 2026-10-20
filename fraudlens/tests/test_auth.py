import sqlite3

import pytest

from fraudkit.auth import ApiKeyStore, AuthError, UserStore, can, hash_password, password_problems, verify_password
from fraudkit.db import Database

FAST = 1_000  # low PBKDF2 iteration count keeps the tests quick; production uses 600,000


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.sqlite")


@pytest.fixture
def users(db):
    return UserStore(db, max_failed=3, lockout_minutes=15, iterations=FAST)


def test_password_hashing_is_salted_and_verifiable():
    a, b = hash_password("Correct horse 42", iterations=FAST), hash_password("Correct horse 42", iterations=FAST)
    assert a != b  # random salt
    assert verify_password("Correct horse 42", a)
    assert not verify_password("correct horse 42", a)
    assert not verify_password("x", "garbage")


def test_password_policy():
    assert password_problems("short1")
    assert password_problems("onlyletterslong")
    assert password_problems("alice12345678", "alice")
    assert not password_problems("Tr0ub4dor&3xyz")


def test_create_and_authenticate(users):
    users.create("alice", "Analyst-pass-2026", "analyst")
    assert users.authenticate("Alice", "Analyst-pass-2026").role == "analyst"  # usernames are case-insensitive
    with pytest.raises(AuthError, match="Invalid username or password"):
        users.authenticate("alice", "wrong-pass-0000")
    with pytest.raises(AuthError, match="Invalid username or password"):
        users.authenticate("nobody", "whatever-123456")  # same message: no user enumeration
    with pytest.raises(AuthError):
        users.create("alice", "Another-pass-2026", "viewer")
    with pytest.raises(AuthError):
        users.create("Bad Name!", "Another-pass-2026", "viewer")
    with pytest.raises(AuthError):
        users.create("bob", "Another-pass-2026", "superuser")


def test_lockout_after_failed_attempts(users):
    users.create("carol", "Viewer-pass-2026", "viewer")
    for _ in range(3):
        with pytest.raises(AuthError):
            users.authenticate("carol", "nope-nope-0000")
    with pytest.raises(AuthError, match="locked"):
        users.authenticate("carol", "Viewer-pass-2026")  # right password, still locked


def test_last_admin_is_protected(users):
    users.create("root", "Admin-pass-2026", "admin")
    with pytest.raises(AuthError):
        users.set_role("root", "viewer", actor="root")
    with pytest.raises(AuthError):
        users.set_active("root", False, actor="someone")
    users.create("ops", "Second-admin-2026", "admin")
    users.set_role("root", "viewer", actor="ops")
    assert users.get("root").role == "viewer"


def test_disabled_user_cannot_login(users):
    users.create("root", "Admin-pass-2026", "admin")
    users.create("dave", "Analyst-two-2026", "analyst")
    users.set_active("dave", False, actor="root")
    with pytest.raises(AuthError, match="disabled"):
        users.authenticate("dave", "Analyst-two-2026")


def test_role_permissions():
    assert can("admin", "retrain") and not can("analyst", "retrain") and not can("viewer", "retrain")
    assert can("analyst", "decide_cases") and not can("viewer", "decide_cases")
    assert can("viewer", "view") and not can(None, "view")


def test_api_keys_hashed_and_revocable(db):
    keys = ApiKeyStore(db)
    rec, token = keys.create("payments-gateway", actor="root")
    assert token.startswith("fl_") and len(token) > 40
    with sqlite3.connect(db.path) as c:  # plaintext never stored
        stored = [r[0] for r in c.execute("SELECT key_hash FROM api_keys")]
    assert token not in stored
    assert keys.verify(token).key_id == rec.key_id
    assert keys.get(rec.key_id).use_count == 1
    assert keys.verify("fl_not-a-real-key") is None
    keys.revoke(rec.key_id, actor="root")
    assert keys.verify(token) is None


def test_admin_actions_are_audited(db, users):
    users.create("root", "Admin-pass-2026", "admin")
    users.authenticate("root", "Admin-pass-2026")
    with sqlite3.connect(db.path) as c:
        actions = [r[0] for r in c.execute("SELECT action FROM admin_audit ORDER BY id")]
    assert actions == ["user_created", "login"]
