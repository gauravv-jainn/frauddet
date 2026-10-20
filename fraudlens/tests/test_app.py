"""Headless click-through of all six Streamlit pages (needs a trained model in artifacts/)."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
READY = all((ART / f).exists() for f in ["model.joblib", "metrics.json", "test_scored.parquet", "history.parquet"])
pytestmark = pytest.mark.skipif(not READY, reason="no trained artifacts")

PAGES = ROOT / "app" / "views"


@pytest.fixture(autouse=True)
def temp_case_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUDLENS_CASE_DB", str(tmp_path / "cases.sqlite"))


def run_page(name, timeout=180):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGES / name), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_dashboard_matches_metrics_json():
    at = run_page("1_Dashboard.py")
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
    got = confusion_at(scored["is_fraud"], scored["p"], scored["amt"], metrics["threshold"]["value"],
                       metrics["threshold"]["review_cost"])
    assert got == metrics["test"]["at_threshold"]


def test_check_transaction_and_whatif():
    at = run_page("2_Check_Transaction.py")
    assert any("Recommended action" in md.value for md in at.markdown)
    at.number_input[0].set_value(9999.0)
    at.button[0].click().run()  # form submit = Re-score
    assert not at.exception, [e.value for e in at.exception]
    assert any("&rarr;" in md.value for md in at.markdown)
    # every list and the manual form
    at.radio[0].set_value("Enter a transaction").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Recommended action" in md.value for md in at.markdown)


def test_batch_scoring_sample():
    at = run_page("3_Batch_Scoring.py")
    btn = next(b for b in at.button if "sample file" in b.label)
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert metric(at, "Rows scored") == "500"


def test_case_queue_flow():
    at = run_page("4_Case_Queue.py")
    next(b for b in at.button if b.label == "Add top 25 by risk").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert str(metric(at, "Cases")) == "25"
    next(b for b in at.button if b.label == "Confirm fraud").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert str(metric(at, "Confirmed")) == "1"
    next(b for b in at.button if b.label == "Start demo stream").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_model_report():
    at = run_page("5_Model_Report.py")
    m = json.loads((ART / "metrics.json").read_text())["test"]
    assert metric(at, "PR-AUC") == f"{m['pr_auc']:.4f}"


def test_training_lab_renders():
    at = run_page("6_Training_Lab.py")
    assert any(b.label == "Retrain" for b in at.button)
