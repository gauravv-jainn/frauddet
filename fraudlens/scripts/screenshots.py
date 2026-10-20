"""Capture README screenshots with Playwright.

Starts its own Streamlit server on a throwaway database with a generated admin account (the
password is random and never printed), signs in through the UI and visits every page.

    python scripts/screenshots.py
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "screenshots"
PORT = 8599
BASE = f"http://localhost:{PORT}"


def settle(page, extra_ms: int = 2500):
    page.wait_for_selector('[data-testid="stApp"]', timeout=60_000)
    page.wait_for_timeout(1500)
    for _ in range(120):  # wait while Streamlit shows its "running" indicator
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(extra_ms)


def shot(page, name: str, full: bool = True):
    """Streamlit scrolls inside its main container, so grow the viewport to the content height first."""
    OUT.mkdir(parents=True, exist_ok=True)
    if full:
        h = page.evaluate("""() => {
            const m = document.querySelector('[data-testid="stMain"]') || document.querySelector('section.main');
            const c = m && (m.querySelector('[data-testid="stMainBlockContainer"]') || m.firstElementChild);
            return Math.ceil((c ? c.scrollHeight : document.body.scrollHeight) + 80);
        }""")
        page.set_viewport_size({"width": 1280, "height": max(900, min(int(h), 6000))})
        page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / f"{name}.png"))
    page.set_viewport_size({"width": 1280, "height": 900})
    print("saved", OUT / f"{name}.png")


def goto(page, title: str, extra_ms: int = 2500):
    page.get_by_role("link", name=title, exact=True).click()  # client-side navigation keeps the session
    settle(page, extra_ms)


def main():
    from fraudkit.auth import UserStore
    from fraudkit.db import Database

    tmp = Path(tempfile.mkdtemp(prefix="fraudlens_shots_"))
    db_path = tmp / "demo.sqlite"
    user, password = "demo.admin", "Demo-" + secrets.token_urlsafe(18) + "9"
    UserStore(Database(db_path)).create(user, password, "admin", actor="screenshots")
    env = {**os.environ, "FRAUDLENS_DB": str(db_path)}
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(PORT)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{BASE}/_stcore/health", timeout=2)  # noqa: S310 - fixed localhost URL
                break
            except OSError:
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
            page.goto(BASE)
            settle(page, 1000)
            shot(page, "0_login", full=False)
            page.get_by_role("textbox", name="Username").fill(user)
            page.get_by_role("textbox", name="Password").fill(password)
            page.get_by_role("button", name="Sign in").click()
            settle(page)
            shot(page, "1_dashboard")

            goto(page, "Check a transaction")
            shot(page, "2_check_transaction")
            page.get_by_label("Amount (€)").fill("35")
            page.get_by_role("button", name="Re-score").click()
            settle(page)
            shot(page, "2b_what_if")

            goto(page, "Batch scoring")
            page.get_by_role("button", name="Or score the 500-row sample file").click()
            settle(page, 4000)
            shot(page, "3_batch_scoring")

            goto(page, "Case queue")
            page.get_by_role("button", name="Add top 25 by risk").click()
            settle(page)
            page.get_by_role("button", name="Assign to me").click()
            settle(page)
            page.get_by_role("button", name="Confirm fraud").click()
            settle(page)
            shot(page, "4_case_queue")

            goto(page, "Monitoring", 5000)
            shot(page, "7_monitoring")
            goto(page, "Model report", 4000)
            shot(page, "5_model_report")
            goto(page, "Training Lab", 4000)
            shot(page, "6_training_lab")
            goto(page, "Admin")
            shot(page, "8_admin")
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)


if __name__ == "__main__":
    main()
