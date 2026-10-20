"""SQLite storage for users, API keys, cases and audit trails.

WAL mode + a busy timeout let several analysts (Streamlit sessions) and the API write at the
same time without "database is locked" errors. Schema changes are numbered migrations.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

MIGRATIONS: list[str] = [
    # 1: cases + case audit (original schema)
    """
    CREATE TABLE IF NOT EXISTS cases (
        case_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        trans_num    TEXT UNIQUE,
        cc_num       TEXT,
        txn_time     TEXT,
        amount       REAL,
        category     TEXT,
        merchant     TEXT,
        score        REAL,
        band         TEXT,
        reasons      TEXT,
        status       TEXT NOT NULL DEFAULT 'open',
        analyst_note TEXT DEFAULT '',
        source       TEXT,
        created_at   TEXT,
        updated_at   TEXT
    );
    CREATE TABLE IF NOT EXISTS audit (
        audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id     INTEGER NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
        action      TEXT NOT NULL,
        from_status TEXT,
        to_status   TEXT,
        note        TEXT,
        actor       TEXT,
        at          TEXT
    );
    """,
    # 2: users, API keys, admin audit, case assignment, model version on cases
    """
    CREATE TABLE IF NOT EXISTS users (
        username        TEXT PRIMARY KEY,
        password_hash   TEXT NOT NULL,
        role            TEXT NOT NULL CHECK (role IN ('viewer', 'analyst', 'admin')),
        active          INTEGER NOT NULL DEFAULT 1,
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        locked_until    TEXT,
        must_change_pw  INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL,
        last_login      TEXT
    );
    CREATE TABLE IF NOT EXISTS api_keys (
        key_id      TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        key_hash    TEXT NOT NULL UNIQUE,
        created_by  TEXT,
        created_at  TEXT NOT NULL,
        last_used   TEXT,
        use_count   INTEGER NOT NULL DEFAULT 0,
        revoked_at  TEXT
    );
    CREATE TABLE IF NOT EXISTS admin_audit (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        at      TEXT NOT NULL,
        actor   TEXT,
        action  TEXT NOT NULL,
        target  TEXT,
        detail  TEXT
    );
    ALTER TABLE cases ADD COLUMN assignee TEXT;
    ALTER TABLE cases ADD COLUMN model_version TEXT;
    CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
    CREATE INDEX IF NOT EXISTS idx_audit_case ON audit(case_id);
    """,
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(path: str | Path) -> int:
    """Apply pending migrations; returns the schema version."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0 and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cases'").fetchone():
            version = 1  # database created before migrations existed
        for i in range(version, len(MIGRATIONS)):
            conn.executescript("BEGIN;" + MIGRATIONS[i] + f"PRAGMA user_version={i + 1};COMMIT;")
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


class Database:
    """Opens a short-lived connection per operation (safe across Streamlit threads)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.schema_version = migrate(self.path)

    @contextmanager
    def tx(self):
        conn = connect(self.path)
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_admin(self, actor: str | None, action: str, target: str | None = None, detail: str | None = None):
        with self.tx() as c:
            c.execute(
                "INSERT INTO admin_audit (at, actor, action, target, detail) VALUES (?,?,?,?,?)",
                (now(), actor, action, target, detail),
            )
