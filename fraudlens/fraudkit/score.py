"""Model bundle: features -> raw score -> calibrated probability -> risk band."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import CATEGORICAL, FEATURES, MerchantEncoder, build_features, featurize_with_history, model_frame

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

BANDS = ["Low", "Medium", "High", "Critical"]
ACTIONS = {
    "Low": "Approve",
    "Medium": "Approve with step-up verification (one-time code to the cardholder)",
    "High": "Hold the transaction and send it to an analyst for review",
    "Critical": "Block the transaction and call the customer",
}


class Calibrator:
    """Maps raw model probabilities to calibrated ones (isotonic or Platt)."""

    def __init__(self, method: str):
        self.method = method
        self.model = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
        else:
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(C=1e6).fit(_logit(p_raw).reshape(-1, 1), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype="float64")
        if self.method == "isotonic":
            return self.model.predict(p_raw)
        return self.model.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


@dataclass
class FraudModel:
    model: object                 # LGBMClassifier
    calibrator: Calibrator
    encoder: MerchantEncoder
    categories: dict
    threshold: float
    bands: dict                   # lower bounds for Medium/High/Critical
    review_cost: float
    model_name: str
    data_source: str
    versions: dict = field(default_factory=dict)
    features: list = field(default_factory=lambda: list(FEATURES))

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
        high = max(self.bands["high"], med)          # keep the bands ordered for any custom threshold
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
    def load(path: Path | None = None) -> "FraudModel":
        return joblib.load(path or ARTIFACTS / "model.joblib")


def load_metrics(path: Path | None = None) -> dict:
    with open(path or ARTIFACTS / "metrics.json", encoding="utf-8") as f:
        return json.load(f)
