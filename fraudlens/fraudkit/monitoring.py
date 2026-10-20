"""Data and score drift monitoring with the Population Stability Index (PSI).

At training time ``reference_profile`` stores, for every feature, the bin edges and bin shares
seen on the fit period. In production ``drift_report`` bins new data with the same edges and
compares shares. Usual reading: PSI < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
SCORE_EDGES = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
_EPS = 1e-4
_OTHER = "__other__"
_MISSING = "__missing__"


def _numeric_shares(values: np.ndarray, edges: list[float]) -> list[float]:
    v = np.asarray(values, dtype="float64")
    missing = np.isnan(v)
    inner = np.asarray(edges, dtype="float64")
    idx = np.searchsorted(inner, v[~missing], side="right")  # bins: (-inf,e0], (e0,e1], ... (e_last, inf)
    counts = np.bincount(idx, minlength=len(inner) + 1).astype("float64")
    counts = np.r_[counts, missing.sum()]
    return (counts / max(len(v), 1)).tolist()


def _categorical_shares(values: pd.Series, levels: list[str]) -> list[float]:
    s = values.astype("object").where(values.notna(), _MISSING).astype(str)
    s = s.where(s.isin(levels), _OTHER)
    counts = s.value_counts()
    total = max(len(s), 1)
    return [float(counts.get(lv, 0)) / total for lv in levels + [_OTHER]]


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, dtype="float64"), _EPS, None)
    a = np.clip(np.asarray(actual, dtype="float64"), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def reference_profile(
    X: pd.DataFrame, numeric: list[str], categorical: list[str], scores=None, n_bins: int = 10
) -> dict:
    """Bin edges and shares of the training data (fit period) for every feature and the score."""
    prof: dict = {"numeric": {}, "categorical": {}, "rows": int(len(X))}
    for c in numeric:
        v = X[c].to_numpy(dtype="float64")
        finite = v[np.isfinite(v)]
        qs = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(finite) else np.array([])
        edges = qs.tolist()
        prof["numeric"][c] = {
            "edges": edges,
            "shares": _numeric_shares(v, edges),
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        }
    for c in categorical:
        levels = sorted(X[c].astype("object").dropna().astype(str).unique().tolist())
        if X[c].isna().any():
            levels.append(_MISSING)
        prof["categorical"][c] = {"levels": levels, "shares": _categorical_shares(X[c], levels)}
    if scores is not None:
        prof["score"] = {"edges": SCORE_EDGES[1:-1], "shares": _numeric_shares(np.asarray(scores), SCORE_EDGES[1:-1])}
    return prof


def training_ranges(profile: dict | None) -> dict:
    """{feature: {"min", "max"}} seen in training, for the out-of-range policy rule."""
    if not profile:
        return {}
    return {c: {"min": v["min"], "max": v["max"]} for c, v in profile["numeric"].items() if v.get("max") is not None}


def drift_report(profile: dict, X: pd.DataFrame, scores=None) -> pd.DataFrame:
    """PSI of every feature (and the score) of ``X`` against the reference profile."""
    rows = []
    for c, ref in profile["numeric"].items():
        if c in X:
            v = psi(ref["shares"], _numeric_shares(X[c].to_numpy(dtype="float64"), ref["edges"]))
            rows.append({"feature": c, "kind": "numeric", "psi": v, "status": status(v)})
    for c, ref in profile["categorical"].items():
        if c in X:
            v = psi(ref["shares"], _categorical_shares(X[c], ref["levels"]))
            rows.append({"feature": c, "kind": "categorical", "psi": v, "status": status(v)})
    if scores is not None and "score" in profile:
        ref = profile["score"]
        v = psi(ref["shares"], _numeric_shares(np.asarray(scores), ref["edges"]))
        rows.append({"feature": "model score", "kind": "score", "psi": v, "status": status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
