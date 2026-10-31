"""Deploy FraudLens to a free Hugging Face Docker Space: python scripts/deploy_space.py

Asks for two tokens from https://huggingface.co/settings/tokens (input is hidden, neither is written to disk):
  * a "Write" token: used only by this script to create and upload the repos.
  * a "Read" token: stored as a Space secret so the running app can download the private artifacts.

The app code goes to a public Space. The model and transaction data go to a private dataset repo, never to GitHub.
Generated sign-in passwords are saved to deploy/credentials.txt (git-ignored). Safe to run again to update.
"""

from __future__ import annotations

import getpass
import secrets
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = "fraudlens"
ARTIFACTS_NAME = "fraudlens-artifacts"
CREDENTIALS = ROOT / "deploy" / "credentials.txt"
CODE_DIRS = ("fraudkit", "app", "configs", ".streamlit")
ROOT_FILES = ("requirements.txt", "requirements-deploy.txt")
ARTIFACT_PATTERNS = ["*.joblib", "*.parquet", "*.jsonl", "*.csv", "*.json", "runs/*"]
ARTIFACT_IGNORE = ["run_status.json", "retrain_stdout.log", ".*.tmp"]


def passwords() -> dict:
    """Reuse saved passwords so re-deploying does not change the sign-in details."""
    if CREDENTIALS.exists():
        rows = (line.split("=", 1) for line in CREDENTIALS.read_text(encoding="utf-8").splitlines() if "=" in line)
        return {k.strip(): v.strip() for k, v in rows}
    # token_urlsafe can lack a digit, which the password policy requires
    creds = {"admin": f"{secrets.token_urlsafe(15)}7", "demo": f"{secrets.token_urlsafe(9)}7"}
    CREDENTIALS.write_text("".join(f"{k} = {v}\n" for k, v in creds.items()), encoding="utf-8")
    return creds


def code_operations() -> list:
    files = [ROOT / f for f in ROOT_FILES]
    for d in CODE_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    ops = [CommitOperationAdd(p.relative_to(ROOT).as_posix(), str(p)) for p in files]
    for name in ("Dockerfile", "README.md"):
        ops.append(CommitOperationAdd(name, str(ROOT / "deploy" / "space" / name)))
    return ops


def main() -> int:
    write_token = getpass.getpass("Hugging Face WRITE token (hidden): ").strip()
    read_token = getpass.getpass("Hugging Face READ token (hidden): ").strip()
    api = HfApi(token=write_token)
    user = api.whoami()["name"]
    space_id, artifacts_id = f"{user}/{SPACE_NAME}", f"{user}/{ARTIFACTS_NAME}"
    creds = passwords()

    print(f"Uploading artifacts to private dataset {artifacts_id}", flush=True)
    api.create_repo(artifacts_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=artifacts_id,
        repo_type="dataset",
        folder_path=ROOT / "artifacts",
        allow_patterns=ARTIFACT_PATTERNS,
        ignore_patterns=ARTIFACT_IGNORE,
        commit_message="Update artifacts",
    )

    print(f"Creating Space {space_id}", flush=True)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(space_id, "HF_TOKEN", read_token)
    api.add_space_secret(space_id, "FRAUDLENS_ADMIN_PASSWORD", creds["admin"])
    api.add_space_secret(space_id, "FRAUDLENS_DEMO_PASSWORD", creds["demo"])
    api.add_space_variable(space_id, "FRAUDLENS_ARTIFACTS_REPO", artifacts_id)
    api.create_commit(space_id, code_operations(), repo_type="space", commit_message="Deploy FraudLens")

    print(f"\nLive URL (ready after the first build): https://{user.lower()}-{SPACE_NAME}.hf.space")
    print(f"Build logs: https://huggingface.co/spaces/{space_id}")
    print(f"Sign-in passwords: {CREDENTIALS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
