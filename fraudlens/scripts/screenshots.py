"""Capture README screenshots of a running FraudLens app with Playwright.

    streamlit run app/streamlit_app.py            (in one terminal)
    python scripts/screenshots.py                 (in another)
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def settle(page, extra_ms: int = 2500):
    page.wait_for_selector('[data-testid="stApp"]', timeout=60_000)
    page.wait_for_timeout(1500)
    # wait while Streamlit shows its "running" indicator
    for _ in range(120):
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


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
        page.goto(f"{BASE}/Dashboard")
        settle(page)
        shot(page, "1_dashboard")

        page.goto(f"{BASE}/Check_Transaction")
        settle(page)
        shot(page, "2_check_transaction")
        amount = page.get_by_label("Amount (EUR)")
        amount.fill("35")
        page.get_by_role("button", name="Re-score").click()
        settle(page)
        shot(page, "2b_what_if")

        page.goto(f"{BASE}/Batch_Scoring")
        settle(page)
        page.get_by_role("button", name="Or score the 500-row sample file").click()
        settle(page, 4000)
        shot(page, "3_batch_scoring")

        page.goto(f"{BASE}/Case_Queue")
        settle(page)
        page.get_by_role("button", name="Add top 25 by risk").click()
        settle(page)
        shot(page, "4_case_queue")

        page.goto(f"{BASE}/Model_Report")
        settle(page, 4000)
        shot(page, "5_model_report")

        page.goto(f"{BASE}/Training_Lab")
        settle(page, 4000)
        shot(page, "6_training_lab")
        browser.close()


if __name__ == "__main__":
    main()
