"""Operator CLI for users and API keys (for servers and containers without a browser).

python -m fraudkit.admin create-user alice --role analyst      (prompts for the password)
python -m fraudkit.admin list-users
python -m fraudkit.admin set-role alice admin
python -m fraudkit.admin disable alice | enable alice
python -m fraudkit.admin reset-password alice                 (prompts; user must change it at sign-in)
python -m fraudkit.admin create-api-key "checkout-service"    (prints the key once)
python -m fraudkit.admin list-api-keys
python -m fraudkit.admin revoke-api-key key_1a2b3c4d
"""

from __future__ import annotations

import argparse
import getpass
import sys

from .auth import ROLES, ApiKeyStore, AuthError, UserStore
from .config import load_config
from .db import Database

ACTOR = "cli"


def _password(confirm: bool = True) -> str:
    pw = getpass.getpass("Password: ")
    if confirm and getpass.getpass("Repeat password: ") != pw:
        raise AuthError("Passwords do not match.")
    return pw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fraudkit.admin", description="Manage FraudLens users and API keys")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create-user")
    p.add_argument("username")
    p.add_argument("--role", choices=ROLES, default="analyst")
    sub.add_parser("list-users")
    p = sub.add_parser("set-role")
    p.add_argument("username")
    p.add_argument("role", choices=ROLES)
    for name in ("disable", "enable", "reset-password"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("create-api-key").add_argument("name")
    sub.add_parser("list-api-keys")
    sub.add_parser("revoke-api-key").add_argument("key_id")
    args = ap.parse_args(argv)

    cfg = load_config()
    db = Database(cfg.db_path())
    users = UserStore(db, cfg.app.max_failed_logins, cfg.app.lockout_minutes)
    keys = ApiKeyStore(db)
    try:
        if args.cmd == "create-user":
            u = users.create(args.username, _password(), args.role, actor=ACTOR)
            print(f"created {u.username} ({u.role})")
        elif args.cmd == "list-users":
            for u in users.list():
                print(
                    f"{u.username:<24} {u.role:<8} {'active' if u.active else 'disabled':<9} last login {u.last_login}"
                )
        elif args.cmd == "set-role":
            users.set_role(args.username, args.role, actor=ACTOR)
            print("ok")
        elif args.cmd in ("disable", "enable"):
            users.set_active(args.username, args.cmd == "enable", actor=ACTOR)
            print("ok")
        elif args.cmd == "reset-password":
            users.set_password(args.username, _password(), actor=ACTOR, must_change=True)
            print("ok; the user must choose a new password at next sign-in")
        elif args.cmd == "create-api-key":
            rec, token = keys.create(args.name, actor=ACTOR)
            print(f"{rec.key_id}  {rec.name}\n\n{token}\n\nStore this key now; it cannot be shown again.")
        elif args.cmd == "list-api-keys":
            for k in keys.list():
                print(f"{k.key_id}  {k.name:<30} {'active' if k.active else 'revoked':<8} used {k.use_count}x")
        elif args.cmd == "revoke-api-key":
            keys.revoke(args.key_id, actor=ACTOR)
            print("revoked")
    except AuthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
