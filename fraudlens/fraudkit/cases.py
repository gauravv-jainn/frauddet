"""Analyst case queue (SQLite) with assignment and a full audit trail.

Only the last four digits of a card number are stored here (data minimisation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .db import Database, now
from .security import card_hash, mask_card

STATUSES = ["open", "confirmed_fraud", "genuine", "escalated"]
ACTION_TO_STATUS = {"Confirm fraud": "confirmed_fraud", "Mark genuine": "genuine", "Escalate": "escalated"}


class CaseStore:
    def __init__(self, db: Database | str | Path):
        self.db = db if isinstance(db, Database) else Database(db)

    def add_case(
        self,
        *,
        trans_num: str,
        cc_num,
        txn_time,
        amount: float,
        category: str,
        merchant: str,
        score: float,
        band: str,
        reasons: list[str],
        source: str = "queue",
        actor: str = "system",
        model_version: str | None = None,
        expected_loss: float | None = None,
        decision: str | None = None,
    ) -> int | None:
        """Insert a flagged transaction. Returns the case id, or None if it is already queued."""
        with self.db.tx() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO cases (trans_num, cc_num, txn_time, amount, category, merchant, score, band,"
                " reasons, status, source, created_at, updated_at, model_version, expected_loss, decision, card_ref)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(trans_num),
                    mask_card(cc_num),
                    str(txn_time),
                    float(amount),
                    str(category),
                    str(merchant),
                    float(score),
                    str(band),
                    json.dumps([str(r) for r in reasons]),
                    "open",
                    source,
                    now(),
                    now(),
                    model_version,
                    None if expected_loss is None else float(expected_loss),
                    decision,
                    card_hash(cc_num),
                ),
            )
            if cur.rowcount == 0:
                return None
            cid = cur.lastrowid
            c.execute(
                "INSERT INTO audit (case_id, action, from_status, to_status, note, actor, at) VALUES (?,?,?,?,?,?,?)",
                (cid, "created", None, "open", f"score {score:.4f} ({band}) via {source}", actor, now()),
            )
            return cid

    def update_status(self, case_id: int, action: str, note: str = "", actor: str = "analyst"):
        if action not in ACTION_TO_STATUS:
            raise ValueError(f"unknown action {action!r}")
        new = ACTION_TO_STATUS[action]
        note = (note or "")[:2000]
        with self.db.tx() as c:
            row = c.execute("SELECT status FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(case_id)
            c.execute(
                "UPDATE cases SET status=?, analyst_note=?, updated_at=?, assignee=COALESCE(assignee, ?) "
                "WHERE case_id=?",
                (new, note, now(), actor, case_id),
            )
            c.execute(
                "INSERT INTO audit (case_id, action, from_status, to_status, note, actor, at) VALUES (?,?,?,?,?,?,?)",
                (case_id, action, row["status"], new, note, actor, now()),
            )

    def assign(self, case_id: int, assignee: str | None, actor: str):
        with self.db.tx() as c:
            row = c.execute("SELECT status, assignee FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(case_id)
            c.execute("UPDATE cases SET assignee=?, updated_at=? WHERE case_id=?", (assignee, now(), case_id))
            c.execute(
                "INSERT INTO audit (case_id, action, from_status, to_status, note, actor, at) VALUES (?,?,?,?,?,?,?)",
                (
                    case_id,
                    "assigned",
                    row["status"],
                    row["status"],
                    f"{row['assignee'] or 'unassigned'} -> {assignee or 'unassigned'}",
                    actor,
                    now(),
                ),
            )

    def cases(self, status: str | None = None, assignee: str | None = None) -> pd.DataFrame:
        where, params = [], []
        if status:
            where.append("status=?")
            params.append(status)
        if assignee:
            where.append("assignee=?")
            params.append(assignee)
        q = "SELECT * FROM cases" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY score DESC, case_id"
        with self.db.tx() as c:
            df = pd.read_sql_query(q, c, params=params or None)
        df["reasons"] = df["reasons"].map(lambda s: json.loads(s) if s else [])
        for col in ("assignee", "model_version", "analyst_note", "decision", "card_ref"):  # NULL -> None, not NaN
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        return df

    def audit(self, case_id: int | None = None) -> pd.DataFrame:
        q = "SELECT * FROM audit" + (" WHERE case_id=?" if case_id is not None else "") + " ORDER BY audit_id DESC"
        with self.db.tx() as c:
            return pd.read_sql_query(q, c, params=(case_id,) if case_id is not None else None)

    def bulk_update(self, case_ids: list[int], action: str, note: str, actor: str) -> int:
        for cid in case_ids:
            self.update_status(int(cid), action, note, actor)
        return len(case_ids)

    def bulk_assign(self, case_ids: list[int], assignee: str | None, actor: str) -> int:
        for cid in case_ids:
            self.assign(int(cid), assignee, actor)
        return len(case_ids)

    def add_comment(self, case_id: int, author: str, text: str) -> int:
        text = (text or "").strip()[:4000]
        if not text:
            raise ValueError("comment is empty")
        with self.db.tx() as c:
            if c.execute("SELECT 1 FROM cases WHERE case_id=?", (case_id,)).fetchone() is None:
                raise KeyError(case_id)
            cur = c.execute(
                "INSERT INTO case_comments (case_id, author, at, text) VALUES (?,?,?,?)", (case_id, author, now(), text)
            )
            c.execute(
                "INSERT INTO audit (case_id, action, from_status, to_status, note, actor, at) "
                "SELECT ?, 'commented', status, status, ?, ?, ? FROM cases WHERE case_id=?",
                (case_id, text[:200], author, now(), case_id),
            )
            return cur.lastrowid

    def comments(self, case_id: int) -> pd.DataFrame:
        with self.db.tx() as c:
            return pd.read_sql_query(
                "SELECT author, at, text FROM case_comments WHERE case_id=? ORDER BY id", c, params=(case_id,)
            )

    def stats(self, sla_hours: float | None = None, user: str | None = None) -> dict:
        df = self.cases()
        counts = df["status"].value_counts().to_dict() if len(df) else {}
        confirmed, genuine = counts.get("confirmed_fraud", 0), counts.get("genuine", 0)
        decided = confirmed + genuine
        return {
            "total": len(df),
            **{s: int(counts.get(s, 0)) for s in STATUSES},
            "decided": int(decided),
            # every case was flagged by the model, so "confirmed fraud" = analyst agrees with the model
            "agreement_rate": confirmed / decided if decided else None,
            **self._sla(df, sla_hours, user),
        }

    @staticmethod
    def _sla(df: pd.DataFrame, sla_hours: float | None, user: str | None) -> dict:
        if not len(df):
            return {"overdue": 0, "my_open": 0, "my_overdue": 0}
        active = df[df["status"].isin(["open", "escalated"])]
        age_h = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(active["created_at"], utc=True)).dt.total_seconds() / 3600
        overdue = age_h > sla_hours if sla_hours else pd.Series(False, index=active.index)
        mine = active["assignee"] == user if user else pd.Series(False, index=active.index)
        return {"overdue": int(overdue.sum()), "my_open": int(mine.sum()), "my_overdue": int((mine & overdue).sum())}

    def clear(self, actor: str = "system"):
        with self.db.tx() as c:
            n = c.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            c.execute("DELETE FROM audit")
            c.execute("DELETE FROM cases")
        self.db.log_admin(actor, "queue_cleared", None, f"{n} cases deleted")
