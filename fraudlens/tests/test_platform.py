"""Config validation, drift monitoring, security helpers, DB migrations and evaluation maths."""

import sqlite3

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.metrics import average_precision_score

from fraudkit.config import Settings, load_config
from fraudkit.db import Database
from fraudkit.monitoring import drift_report, psi, reference_profile
from fraudkit.security import atomic_write_text, esc, mask_card, sanitize_csv_frame
from fraudkit.train import average_precision_weighted


def test_default_config_is_valid():
    cfg = load_config()
    assert cfg.bands.high < cfg.bands.critical
    assert cfg.model.optuna_trials >= 1


@pytest.mark.parametrize(
    "bad",
    [
        {"bands": {"high": 0.8, "critical": 0.7}},
        {"bands": {"medium": 0.5, "high": 0.3}},
        {"costs": {"review_cost": -1}},
        {"data": {"validation_fraction": 0.9}},
        {"model": {"optuna_trials": 0}},
        {"unknown_section": {}},
    ],
)
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValidationError):
        Settings.model_validate(bad)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 20_000), "c": rng.choice(["a", "b"], 20_000)})
    prof = reference_profile(ref, ["x"], ["c"])
    same = pd.DataFrame({"x": rng.normal(0, 1, 20_000), "c": rng.choice(["a", "b"], 20_000)})
    shifted = pd.DataFrame({"x": rng.normal(1.5, 1, 20_000), "c": rng.choice(["a", "b", "zzz"], 20_000)})
    r_same = drift_report(prof, same).set_index("feature")
    r_shift = drift_report(prof, shifted).set_index("feature")
    assert r_same.loc["x", "psi"] < 0.01 and r_same.loc["x", "status"] == "stable"
    assert r_shift.loc["x", "psi"] > 0.25 and r_shift.loc["x", "status"] == "significant"
    assert r_shift.loc["c", "psi"] > r_same.loc["c", "psi"]  # unseen category counted as drift
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_esc_and_csv_injection():
    assert esc("<img src=x onerror=alert(1)>") == "&lt;img src=x onerror=alert(1)&gt;"
    df = pd.DataFrame({"merchant": ['=HYPERLINK("http://evil")', "+1", "@SUM(A1)", "Acme"], "amt": [-1.0, 2, 3, 4]})
    out = sanitize_csv_frame(df)
    assert out["merchant"].tolist() == ['\'=HYPERLINK("http://evil")', "'+1", "'@SUM(A1)", "Acme"]
    assert out["amt"].tolist() == [-1.0, 2, 3, 4]  # numbers untouched


def test_mask_card():
    assert mask_card(4000123412341234) == "****1234"


def test_atomic_write(tmp_path):
    p = tmp_path / "f.json"
    atomic_write_text(p, "one")
    atomic_write_text(p, "two")
    assert p.read_text() == "two"
    assert [x.name for x in tmp_path.iterdir()] == ["f.json"]  # no temp files left behind


def test_migration_upgrades_old_case_database(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as c:  # schema written by the first version of the app
        c.executescript(
            "CREATE TABLE cases (case_id INTEGER PRIMARY KEY AUTOINCREMENT, trans_num TEXT UNIQUE,"
            " cc_num TEXT, txn_time TEXT, amount REAL, category TEXT, merchant TEXT, score REAL,"
            " band TEXT, reasons TEXT, status TEXT NOT NULL DEFAULT 'open', analyst_note TEXT DEFAULT '',"
            " source TEXT, created_at TEXT, updated_at TEXT);"
            "CREATE TABLE audit (audit_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER NOT NULL,"
            " action TEXT NOT NULL, from_status TEXT, to_status TEXT, note TEXT, actor TEXT, at TEXT);"
            "INSERT INTO cases (trans_num, status) VALUES ('t1', 'open');"
        )
    db = Database(path)
    assert db.schema_version == 2
    with db.tx() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(cases)")}
        assert {"assignee", "model_version"} <= cols
        assert c.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 1  # data kept


def test_weighted_average_precision_matches_sklearn():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(float)
    p = np.round(np.clip(y * 0.3 + rng.random(5000) * 0.7, 0, 1), 2)  # rounding creates ties
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ends = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
    assert average_precision_weighted(y[order], ends, np.ones_like(y)) == pytest.approx(
        average_precision_score(y, p), abs=1e-12
    )
