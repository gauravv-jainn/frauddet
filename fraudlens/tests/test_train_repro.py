"""Training twice with the same config must give identical metrics (uses --fast on a small card sample)."""
import json
from pathlib import Path

import pytest

from fraudkit import train


def _dataset_available() -> bool:
    try:
        from fraudkit.data import locate_dataset
        return (locate_dataset() / "fraudTrain.csv").exists()
    except Exception:
        return False


def _round(o, nd=10):
    """Parallel tree averaging can differ in the last float bit (1e-16); compare to 10 decimals."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def _comparable(m: dict) -> dict:
    return _round({
        "leaderboard": [{k: r[k] for k in ("model", "pr_auc", "roc_auc", "brier")} for r in m["leaderboard"]],
        "best_params": m["tuning"]["best_params"],
        "threshold": m["threshold"]["value"],
        "test": {k: m["test"][k] for k in ("pr_auc", "pr_auc_ci95", "roc_auc", "brier_after_calibration")},
        "at_threshold": m["test"]["at_threshold"],
    })


@pytest.mark.skipif(not _dataset_available(), reason="Kaggle dataset not cached")
def test_training_is_reproducible(tmp_path: Path):
    args = ["--fast", "--sample-cards", "40", "--trials", "2", "--no-live"]
    train.main(args + ["--out", str(tmp_path / "a")])
    train.main(args + ["--out", str(tmp_path / "b")])
    a = json.loads((tmp_path / "a" / "metrics.json").read_text())
    b = json.loads((tmp_path / "b" / "metrics.json").read_text())
    assert _comparable(a) == _comparable(b)
