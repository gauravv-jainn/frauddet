"""Watchlists: blocked cards (stored as SHA-256 hashes) and watched merchants.

Entries feed the policy rules (fraudkit/rules.py): a blocked card makes later transactions Critical (block),
a watched merchant makes them High (analyst review). Every change is written to the admin audit log.
"""

from __future__ import annotations

import pandas as pd

from .db import Database, now
from .security import card_hash, mask_card

KINDS = ("card", "merchant")


class WatchlistStore:
    def __init__(self, db: Database):
        self.db = db

    def add_card(self, cc_num, reason: str, actor: str) -> bool:
        return self._add("card", card_hash(cc_num), mask_card(cc_num), reason, actor)

    def add_card_ref(self, card_ref: str, label: str, reason: str, actor: str) -> bool:
        """Block a card known only by its hash (e.g. from a case)."""
        return self._add("card", card_ref, label, reason, actor)

    def add_merchant(self, merchant: str, reason: str, actor: str) -> bool:
        merchant = str(merchant).strip()
        if not merchant:
            raise ValueError("merchant name is empty")
        return self._add("merchant", merchant, merchant, reason, actor)

    def _add(self, kind: str, value: str, label: str, reason: str, actor: str) -> bool:
        """Returns True if a new active entry was created (False if it was already active)."""
        with self.db.tx() as c:
            row = c.execute("SELECT id, active FROM watchlist WHERE kind=? AND value=?", (kind, value)).fetchone()
            if row and row["active"]:
                return False
            if row:
                c.execute(
                    "UPDATE watchlist SET active=1, reason=?, added_by=?, added_at=? WHERE id=?",
                    (reason, actor, now(), row["id"]),
                )
            else:
                c.execute(
                    "INSERT INTO watchlist (kind, value, label, reason, added_by, added_at) VALUES (?,?,?,?,?,?)",
                    (kind, value, label, reason, actor, now()),
                )
        self.db.log_admin(actor, f"watchlist_add_{kind}", label, reason)
        return True

    def remove(self, entry_id: int, actor: str):
        with self.db.tx() as c:
            row = c.execute("SELECT kind, label FROM watchlist WHERE id=? AND active=1", (entry_id,)).fetchone()
            if row is None:
                raise KeyError(entry_id)
            c.execute("UPDATE watchlist SET active=0 WHERE id=?", (entry_id,))
        self.db.log_admin(actor, f"watchlist_remove_{row['kind']}", row["label"])

    def remove_card(self, cc_num, actor: str) -> bool:
        """Unblock a card by its hash (never by the masked label: two cards can share the last four digits)."""
        with self.db.tx() as c:
            row = c.execute(
                "SELECT id FROM watchlist WHERE kind='card' AND value=? AND active=1", (card_hash(cc_num),)
            ).fetchone()
        if row is None:
            return False
        self.remove(int(row["id"]), actor)
        return True

    def entries(self, active_only: bool = True) -> pd.DataFrame:
        q = "SELECT id, kind, label, reason, added_by, added_at, active FROM watchlist"
        q += " WHERE active=1" if active_only else ""
        with self.db.tx() as c:
            return pd.read_sql_query(q + " ORDER BY added_at DESC", c)

    def active_sets(self) -> dict:
        """{"cards": {sha256...}, "merchants": {name...}} for the policy rules."""
        with self.db.tx() as c:
            rows = c.execute("SELECT kind, value FROM watchlist WHERE active=1").fetchall()
        return {
            "cards": {r["value"] for r in rows if r["kind"] == "card"},
            "merchants": {r["value"] for r in rows if r["kind"] == "merchant"},
        }

    def is_card_blocked(self, cc_num) -> bool:
        return card_hash(cc_num) in self.active_sets()["cards"]
