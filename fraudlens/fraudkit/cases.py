"""Analyst case queue stored in SQLite, with a full audit trail."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STATUSES = ["open", "confirmed_fraud", "genuine", "escalated"]
ACTION_TO_STATUS = {"Confirm fraud": "confirmed_fraud", "Mark genuine": "genuine", "Escalate": "escalated"}

SCHEMA = """
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
    reasons      TEXT,      -- JSON list of reason sentences (raising first)
    status       TEXT NOT NULL DEFAULT 'open',
    analyst_note TEXT DEFAULT '',
    source       TEXT,      -- 'queue' | 'demo' | 'batch' | 'manual'
    created_at   TEXT,
    updated_at   TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     INTEGER NOT NULL,
    action      TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT,
    note        TEXT,
    actor       TEXT,
    at          TEXT,
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaseStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_case(self, *, trans_num: str, cc_num, txn_time, amount: float, category: str, merchant: str,
                 score: float, band: str, reasons: list[str], source: str = "queue", actor: str = "system") -> int | None:
        """Insert a flagged transaction. Returns the case id, or None if it is already queued."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO cases (trans_num, cc_num, txn_time, amount, category, merchant, score, band,"
                " reasons, status, source, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(trans_num), str(cc_num), str(txn_time), float(amount), category, merchant, float(score), band,
                 json.dumps(reasons), "open", source, _now(), _now()))
            if cur.rowcount == 0:
                return None
            cid = cur.lastrowid
            c.execute("INSERT INTO audit (case_id, action, from_status, to_status, note, actor, at) VALUES (?,?,?,?,?,?,?)",
                      (cid, "created", None, "open", f"score {score:.4f} ({band})", actor, _now()))
            return cid

    def update_status(self, case_id: int, action: str, note: str = "", actor: str = "analyst"):
        if action not in ACTION_TO_STATUS:
            raise ValueError(f"unknown action {action!r}")
        new = ACTION_TO_STATUS[action]
        with self._conn() as c:
            row = c.execute("SELECT status FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(case_id)
            c.execute("UPDATE cases SET status=?, analyst_note=?, updated_at=? WHERE case_id=?",
                      (new, note, _now(), case_id))
            c.execute("INSERT INTO audit (case_id, action, from_status, to_status, note, actor, at) VALUES (?,?,?,?,?,?,?)",
                      (case_id, action, row[0], new, note, actor, _now()))

    def cases(self, status: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM cases" + (" WHERE status=?" if status else "") + " ORDER BY score DESC, case_id"
        with self._conn() as c:
            df = pd.read_sql_query(q, c, params=(status,) if status else None)
        df["reasons"] = df["reasons"].map(lambda s: json.loads(s) if s else [])
        return df

    def audit(self, case_id: int | None = None) -> pd.DataFrame:
        q = "SELECT * FROM audit" + (" WHERE case_id=?" if case_id is not None else "") + " ORDER BY audit_id DESC"
        with self._conn() as c:
            return pd.read_sql_query(q, c, params=(case_id,) if case_id is not None else None)

    def stats(self) -> dict:
        df = self.cases()
        counts = df["status"].value_counts().to_dict() if len(df) else {}
        confirmed, genuine = counts.get("confirmed_fraud", 0), counts.get("genuine", 0)
        decided = confirmed + genuine
        return {
            "total": int(len(df)),
            **{s: int(counts.get(s, 0)) for s in STATUSES},
            "decided": int(decided),
            # every case was flagged by the model, so "confirmed fraud" = analyst agrees with the model
            "agreement_rate": confirmed / decided if decided else None,
        }

    def clear(self):
        with self._conn() as c:
            c.execute("DELETE FROM audit")
            c.execute("DELETE FROM cases")
