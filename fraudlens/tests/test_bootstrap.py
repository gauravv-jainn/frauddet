"""Hosted start-up: a public URL must never reach the first-run setup screen."""

from fraudkit import bootstrap
from fraudkit.auth import UserStore
from fraudkit.db import Database


def test_refuses_to_start_without_passwords(monkeypatch, capsys):
    monkeypatch.delenv("FRAUDLENS_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("FRAUDLENS_DEMO_PASSWORD", "guest-password-123")
    assert bootstrap.main() == 1
    assert "must be set" in capsys.readouterr().err


def test_accounts_are_created_once_with_the_right_roles(tmp_path):
    users = UserStore(Database(tmp_path / "t.sqlite"))
    bootstrap.ensure_accounts(users, "first-boss-pass-1", "first-guest-pass-1")
    bootstrap.ensure_accounts(users, "other-boss-pass-2", "other-guest-pass-2")  # restart: nothing changes
    assert users.count() == 2
    assert users.get("admin").role == "admin" and users.get("demo").role == "analyst"
    assert not users.get("demo").must_change_pw
    assert users.authenticate("demo", "first-guest-pass-1").username == "demo"
