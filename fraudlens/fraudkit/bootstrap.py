"""Prepare a fresh hosted container before the app starts: fetch the artifacts and create the accounts.

Free hosting has no persistent disk, so this runs on every start. It refuses to start without passwords: otherwise
the app's first-run setup screen would let the first visitor of a public URL become administrator.

Environment: FRAUDLENS_ADMIN_PASSWORD, FRAUDLENS_DEMO_PASSWORD (required), FRAUDLENS_ARTIFACTS_REPO, HF_TOKEN.
"""

from __future__ import annotations

import os
import sys

from .auth import UserStore
from .config import load_config
from .db import Database
from .score import ARTIFACTS

ADMIN_USER, DEMO_USER = "admin", "demo"
ARTIFACT_FILES = ("model.joblib", "metrics.json", "test_scored.parquet", "history.parquet")


def fetch_artifacts(repo: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=ARTIFACTS)


def ensure_accounts(users: UserStore, admin_password: str, demo_password: str) -> None:
    """Create the admin and the demo analyst if missing (a restart with an existing database changes nothing)."""
    for name, password, role in ((ADMIN_USER, admin_password, "admin"), (DEMO_USER, demo_password, "analyst")):
        if not users.get(name):
            users.create(name, password, role, actor="bootstrap")


def main() -> int:
    admin_password = os.environ.get("FRAUDLENS_ADMIN_PASSWORD")
    demo_password = os.environ.get("FRAUDLENS_DEMO_PASSWORD")
    if not admin_password or not demo_password:
        print("FRAUDLENS_ADMIN_PASSWORD and FRAUDLENS_DEMO_PASSWORD must be set.", file=sys.stderr)
        return 1
    repo = os.environ.get("FRAUDLENS_ARTIFACTS_REPO")
    if repo and not all((ARTIFACTS / f).exists() for f in ARTIFACT_FILES):
        print(f"Downloading artifacts from {repo}", flush=True)
        fetch_artifacts(repo, os.environ.get("HF_TOKEN"))
    missing = [f for f in ARTIFACT_FILES if not (ARTIFACTS / f).exists()]
    if missing:
        print(f"Missing artifacts: {', '.join(missing)}", file=sys.stderr)
        return 1
    cfg = load_config()
    users = UserStore(Database(cfg.db_path()), cfg.app.max_failed_logins, cfg.app.lockout_minutes)
    ensure_accounts(users, admin_password, demo_password)
    print("Bootstrap complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
