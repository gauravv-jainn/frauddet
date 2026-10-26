"""Model bundle: features -> raw score -> calibrated probability -> risk band."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, MerchantEncoder, build_features, featurize_with_history, model_frame

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

BANDS = ["Low", "Medium", "High", "Critical"]
ACTIONS = {
    "Low": "Approve",
    "Medium": "Approve with step-up verification (one-time code to the cardholder)",
    "High": "Hold the transaction and send it to an analyst for review",
    "Critical": "Block the transaction and call the customer",
}


P_MIN, P_MAX = 1e-12, 1 - 1e-12  # never exactly 0 or 1, but low enough not to merge tiny scores


SCORE_P_LOW, SCORE_P_HIGH = 1e-6, 0.999


def risk_score(p) -> np.ndarray:
    lo, hi = _logit(SCORE_P_LOW), _logit(SCORE_P_HIGH)
    z = _logit(np.atleast_1d(np.asarray(p, dtype="float64")))
    return np.clip(np.round((z - lo) / (hi - lo) * 999), 0, 999).astype(int)


def score_points_to_double_odds() -> float:
    return float(999 * np.log(2) / (_logit(SCORE_P_HIGH) - _logit(SCORE_P_LOW)))


class Calibrator:
    """Maps raw model probabilities to calibrated ones.

    * ``isotonic``: monotone step function. Flexible, but flat at the ends, so most transactions get the same
      value (exactly 0 before clipping) and ranking resolution is lost.
    * ``platt``: logistic regression on logit(raw). Smooth and strictly monotone (keeps the full ranking).
    * ``beta``: logistic regression on [log(raw), -log(1 - raw)] (Kull et al., 2017). Smooth, strictly monotone
      in practice, and more flexible than Platt for skewed scores.
    Training picks the method with the lowest out-of-sample log-loss on the validation period.
    """

    METHODS = ("isotonic", "platt", "beta")

    def __init__(self, method: str):
        if method not in self.METHODS:
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.model = None

    @staticmethod
    def _design(p_raw: np.ndarray, method: str) -> np.ndarray:
        p = np.clip(np.asarray(p_raw, dtype="float64"), 1e-15, 1 - 1e-15)
        if method == "platt":
            return np.log(p / (1 - p)).reshape(-1, 1)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
        else:
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._design(p_raw, self.method), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype="float64")
        if self.method == "isotonic":
            out = self.model.predict(p_raw)
        else:
            out = self.model.predict_proba(self._design(p_raw, self.method))[:, 1]
        return np.clip(out, P_MIN, P_MAX)


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


@dataclass
class FraudModel:
    model: object  # LGBMClassifier
    calibrator: Calibrator
    encoder: MerchantEncoder
    categories: dict
    threshold: float
    bands: dict  # lower bounds for Medium/High/Critical
    review_cost: float
    model_name: str
    data_source: str
    versions: dict = field(default_factory=dict)
    features: list = field(default_factory=lambda: list(FEATURES))
    version: str = "unversioned"  # set by the trainer: <finished UTC>-<sha256 of model file>
    score_ref: list = field(default_factory=list)  # unused (kept so older model files still load)

    def risk_score(self, p) -> np.ndarray:
        """Risk score 0-999 on a log-odds scale (like credit / fraud scores): p = 1e-6 -> 0, p = 0.999 -> 999,
        and every ~33 points doubles the odds of fraud. Decisions use the probability; the score is for reading."""
        return risk_score(p)

    # --- feature preparation -------------------------------------------------
    def prepare(self, feat: pd.DataFrame) -> pd.DataFrame:
        feat = feat.copy()
        feat["merchant_te"] = self.encoder.transform(feat["merchant"])
        return model_frame(feat, self.categories)

    def featurize(self, rows: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        feat = build_features(rows) if history is None else featurize_with_history(rows, history)
        return feat

    # --- scoring -------------------------------------------------------------
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw(X))

    def band_bounds(self, threshold: float | None = None) -> dict:
        """Medium starts at the decision threshold unless the config pins it."""
        thr = self.threshold if threshold is None else threshold
        med = self.bands.get("medium")
        med = thr if med is None else med
        high = max(self.bands["high"], med)  # keep the bands ordered for any custom threshold
        critical = max(self.bands["critical"], high)
        return {"medium": med, "high": high, "critical": critical}

    def band(self, p, threshold: float | None = None) -> np.ndarray:
        b = self.band_bounds(threshold)
        p = np.atleast_1d(np.asarray(p, dtype="float64"))
        out = np.full(p.shape, "Low", dtype=object)
        out[p >= b["medium"]] = "Medium"
        out[p >= b["high"]] = "High"
        out[p >= b["critical"]] = "Critical"
        return out

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | None = None) -> FraudModel:
        return joblib.load(path or ARTIFACTS / "model.joblib")


def load_serving_model(path: Path | None = None, bands: dict | None = None) -> FraudModel:
    """Model as served by the app and the API: band boundaries come from the current config.

    Both entry points must use this so a transaction gets the same band everywhere.
    """
    fm = FraudModel.load(path)
    if bands is None:
        from .config import load_config

        bands = load_config().bands.model_dump()
    fm.bands = dict(bands)
    return fm


def load_metrics(path: Path | None = None) -> dict:
    with open(path or ARTIFACTS / "metrics.json", encoding="utf-8") as f:
        return json.load(f)
