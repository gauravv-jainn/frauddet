"""Headless click-through of every Streamlit page, per role (needs a trained model in artifacts/)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
READY = all((ART / f).exists() for f in ["model.joblib", "metrics.json", "test_scored.parquet", "history.parquet"])
pytestmark = pytest.mark.skipif(not READY, reason="no trained artifacts")

VIEWS = ROOT / "app" / "views"
PASSWORDS = {"admin": "Admin-test-pass-1", "analyst": "Analyst-test-pass-1", "viewer": "Viewer-test-pass-1"}
USERS = {"admin": "root", "analyst": "kim", "viewer": "vic"}


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUDLENS_DB", str(tmp_path / "fraudlens.sqlite"))
    return tmp_path / "fraudlens.sqlite"


@pytest.fixture
def users(temp_db):
    import sys

    sys.path.insert(0, str(ROOT))
    from fraudkit.auth import UserStore
    from fraudkit.db import Database

    store = UserStore(Database(temp_db), iterations=1_000)
    for role, name in USERS.items():
        store.create(name, PASSWORDS[role], role, actor="test")
    return store


def app_test(path: Path, timeout=180):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(path), default_timeout=timeout)


def run_page(name, role="admin"):
    at = app_test(VIEWS / name)
    now = datetime.now(UTC).isoformat()
    at.session_state["auth"] = {
        "username": USERS[role],
        "role": role,
        "login_at": now,
        "last_seen": now,
        "must_change_pw": False,
    }
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def button(at, label):
    return next(b for b in at.button if b.label == label)


# ----------------------------------------------------------------------------- auth flow
def test_first_run_setup_creates_admin(temp_db):
    at = app_test(ROOT / "app" / "streamlit_app.py")
    at.run()
    assert not at.exception
    assert any("First-run setup" in m.value for m in at.markdown)
    at.text_input[0].set_value("owner")
    at.text_input[1].set_value("First-admin-2026x")
    at.text_input[2].set_value("First-admin-2026x")
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["auth"]["username"] == "owner"
    assert at.session_state["auth"]["role"] == "admin"


def test_login_required_and_wrong_password_rejected(users):
    at = app_test(ROOT / "app" / "streamlit_app.py")
    at.run()
    assert any("Sign in" in m.value for m in at.markdown)
    at.text_input[0].set_value("kim")
    at.text_input[1].set_value("wrong-password-1")
    at.button[0].click().run()
    assert "auth" not in at.session_state
    assert any("Invalid username or password" in e.value for e in at.error)
    at.text_input[1].set_value(PASSWORDS["analyst"])
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["auth"]["role"] == "analyst"


def test_pages_refuse_anonymous_users(users):
    at = app_test(VIEWS / "1_Dashboard.py")
    at.run()
    assert not at.metric  # nothing rendered behind the login


def test_expired_session_is_logged_out(users):
    at = app_test(VIEWS / "1_Dashboard.py")
    at.session_state["auth"] = {
        "username": "kim",
        "role": "analyst",
        "login_at": "2020-01-01T00:00:00+00:00",
        "last_seen": "2020-01-01T00:00:00+00:00",
        "must_change_pw": False,
    }
    at.run()
    assert "auth" not in at.session_state
    assert not at.metric


# ----------------------------------------------------------------------------- pages
def test_dashboard_matches_metrics_json(users):
    at = run_page("1_Dashboard.py", "viewer")
    m = json.loads((ART / "metrics.json").read_text())["test"]["at_threshold"]
    assert metric(at, "Transactions scored") == f"{m['transactions']:,}"
    assert metric(at, "Flagged") == f"{m['flagged']:,}"
    assert metric(at, "Precision") == f"{m['precision']:.1%}"
    assert metric(at, "Recall") == f"{m['recall']:.1%}"
    assert metric(at, "Net saved") == f"€{m['saved_vs_no_model'] / 1e6:.2f}M"
    assert metric(at, "False alarms /10k") == f"{m['false_alarms_per_10k_genuine']:.1f}"


def test_app_numbers_equal_metrics_exactly():
    import sys

    sys.path.insert(0, str(ROOT))
    from fraudkit.costs import confusion_at

    metrics = json.loads((ART / "metrics.json").read_text())
    scored = pd.read_parquet(ART / "test_scored.parquet")
    got = confusion_at(
        scored["is_fraud"],
        scored["p"],
        scored["amt"],
        metrics["threshold"]["value"],
        metrics["threshold"]["review_cost"],
    )
    assert got == metrics["test"]["at_threshold"]


def test_check_transaction_whatif_and_manual(users):
    at = run_page("2_Check_Transaction.py", "analyst")
    assert any("Recommended action" in md.value for md in at.markdown)
    at.number_input[0].set_value(9999.0)
    button(at, "Re-score").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("&rarr;" in md.value for md in at.markdown)
    at.radio[0].set_value("Enter a transaction").run()
    button(at, "Score transaction").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Recommended action" in md.value for md in at.markdown)


def test_viewer_is_read_only_on_check_page(users):
    at = run_page("2_Check_Transaction.py", "viewer")
    assert at.radio[0].options == ["Pick a real test transaction"]
    assert not [b for b in at.button if b.label == "Re-score"]


def test_batch_scoring_sample_and_role(users):
    at = run_page("3_Batch_Scoring.py", "analyst")
    button(at, "Or score the 500-row sample file").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert metric(at, "Rows scored") == "500"
    viewer = run_page("3_Batch_Scoring.py", "viewer")
    assert not [b for b in viewer.button if "sample" in b.label]
    assert any("does not have access" in i.value for i in viewer.info)


def test_case_queue_flow_records_actor(users, temp_db):
    at = run_page("4_Case_Queue.py", "analyst")
    button(at, "Add top 25 by risk").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert str(metric(at, "Cases")) == "25"
    button(at, "Assign to me").click().run()
    button(at, "Confirm fraud").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert str(metric(at, "Confirmed")) == "1"
    import sqlite3

    with sqlite3.connect(temp_db) as c:
        actors = {r[0] for r in c.execute("SELECT actor FROM audit WHERE action='Confirm fraud'")}
        stored_cards = [r[0] for r in c.execute("SELECT cc_num FROM cases")]
    assert actors == {"kim"}
    assert all(s.startswith("****") and len(s) == 8 for s in stored_cards)  # only last 4 digits stored
    assert not [b for b in at.button if b.label == "Delete all cases"]  # analysts cannot clear the queue
    button(at, "Start demo stream").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_viewer_cannot_change_cases(users):
    at = run_page("4_Case_Queue.py", "analyst")
    button(at, "Add top 25 by risk").click().run()
    viewer = run_page("4_Case_Queue.py", "viewer")
    labels = {b.label for b in viewer.button}
    assert not labels & {"Confirm fraud", "Mark genuine", "Escalate", "Add top 25 by risk", "Start demo stream"}


def test_monitoring_page(users):
    run_page("7_Monitoring.py", "viewer")


def test_model_report(users):
    at = run_page("5_Model_Report.py", "viewer")
    m = json.loads((ART / "metrics.json").read_text())["test"]
    assert metric(at, "PR-AUC") == f"{m['pr_auc']:.4f}"


def test_training_lab_retrain_is_admin_only(users):
    admin = run_page("6_Training_Lab.py", "admin")
    analyst = run_page("6_Training_Lab.py", "analyst")
    assert not button(admin, "Retrain").disabled or "running" in json.dumps(admin.session_state.filtered_state)
    assert button(analyst, "Retrain").disabled


def test_admin_creates_user_and_api_key(users):
    at = run_page("8_Admin.py", "admin")
    form_inputs = [t for t in at.text_input if t.label in ("Username", "Temporary password")]
    form_inputs[0].set_value("newbie")
    form_inputs[1].set_value("Temp-pass-20261")
    button(at, "Create user").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert users.get("newbie").must_change_pw
    next(t for t in at.text_input if t.label.startswith("Key name")).set_value("checkout-service")
    button(at, "Create API key").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(c.value.startswith("fl_") for c in at.code)


def test_admin_page_denied_to_analyst(users):
    at = run_page("8_Admin.py", "analyst")
    assert any("does not have access" in i.value for i in at.info)
    assert not at.tabs


def test_account_page(users):
    at = run_page("9_Account.py", "viewer")
    assert metric(at, "Role") == "viewer"


def test_sidebar_settings_survive_page_switches(users):
    """Regression: Streamlit drops widget state when navigation re-creates a widget; the settings must persist."""
    at = run_page("1_Dashboard.py", "analyst")
    cost = next(n for n in at.number_input if n.label.startswith("Review cost"))
    cost.set_value(12.0).run()
    at.checkbox[0].uncheck().run()
    at.session_state["w_review_cost"] = 0.0  # what a page switch does: the widget comes back at its default
    at.session_state["w_use_validated"] = True
    at.run()
    assert not at.exception
    assert next(n for n in at.number_input if n.label.startswith("Review cost")).value == 12.0
    assert at.checkbox[0].value is False
    assert any("Custom threshold or review cost" in c.value for c in at.caption)
