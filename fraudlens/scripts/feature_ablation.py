"""Validation-only ablation of candidate features. Writes artifacts/feature_ablation.json.

Same split, merchant encoder, negative subsampling and tuned hyper-parameters as the last training run; each
candidate set is trained with two seeds so the seed-to-seed noise is visible. The test set is not used.

    python scripts/feature_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from fraudkit.config import load_config
from fraudkit.data import load_raw
from fraudkit.features import CATEGORICAL, NUMERIC, RESEARCH_FEATURES, MerchantEncoder, build_features
from fraudkit.train import lgbm_base, subsample_negatives

CANDIDATES = {
    "current feature set": [],
    "+ amount vs card's previous maximum": ["amt_to_card_max"],
    "+ share of the card's purchases in this category": ["cat_share"],
    "+ how usual this hour is for the card": ["hour_share"],
    **{f"+ {name}": cols for name, cols in RESEARCH_FEATURES.items()},
    "+ all research groups": sum(RESEARCH_FEATURES.values(), []),
    "+ all research groups except weekend/hour": sum(
        (v for k, v in RESEARCH_FEATURES.items() if k != "weekend + circular hour"), []
    ),
    "+ spending windows + z-score + card testing": RESEARCH_FEATURES["spending windows (RFM)"]
    + RESEARCH_FEATURES["amount z-score vs card history"]
    + RESEARCH_FEATURES["card-testing probes"],
}
SEEDS = (42, 7)


def main():
    cfg = load_config()
    m = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    bp = m["tuning"]["best_params"]
    raw = load_raw(cfg.data.cache_dir)
    tr = build_features(raw.train).sort_values("trans_date_trans_time", kind="mergesort")
    cut = tr["trans_date_trans_time"].iloc[int(len(tr) * (1 - cfg.data.validation_fraction))]
    fit, val = tr[tr.trans_date_trans_time < cut].copy(), tr[tr.trans_date_trans_time >= cut].copy()
    enc = MerchantEncoder(smoothing=cfg.features.te_smoothing, seed=cfg.seed)
    fit["merchant_te"] = enc.fit_transform_oof(fit["merchant"], fit["is_fraud"])
    val["merchant_te"] = enc.transform(val["merchant"])
    cats = {c: sorted(fit[c].dropna().unique().tolist()) for c in CATEGORICAL}

    def frame(d, cols):
        X = d[cols].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype("object"), categories=cats[c])
        return X

    y_fit, y_val = fit.is_fraud.to_numpy(), val.is_fraud.to_numpy()
    idx = subsample_negatives(np.arange(len(y_fit)), y_fit, cfg.model.tuning_sample_negatives, cfg.seed)
    rows = []
    for name, extra in CANDIDATES.items():
        cols = NUMERIC + extra + CATEGORICAL
        scores = []
        for seed in SEEDS:
            mdl = LGBMClassifier(**bp, subsample_freq=1, **lgbm_base(seed)).fit(frame(fit, cols).iloc[idx], y_fit[idx])
            scores.append(float(average_precision_score(y_val, mdl.predict_proba(frame(val, cols))[:, 1])))
        rows.append(
            {"candidate": name, "extra_features": extra, "val_pr_auc_by_seed": scores, "mean": float(np.mean(scores))}
        )
        print(f"{name:<52} {scores[0]:.4f} / {scores[1]:.4f}  mean {np.mean(scores):.4f}", flush=True)
    base = rows[0]["mean"]
    noise = abs(rows[0]["val_pr_auc_by_seed"][0] - rows[0]["val_pr_auc_by_seed"][1])
    out = {
        "model_version": m["run"]["model_version"],
        "seeds": list(SEEDS),
        "metric": "validation PR-AUC",
        "seed_noise_baseline": noise,
        "rows": rows,
        "kept": [r["candidate"] for r in rows[1:] if r["mean"] - base > 2 * max(noise, 1e-4)],
    }
    (ROOT / "artifacts" / "feature_ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("kept:", out["kept"] or "none (no gain beyond seed noise)")


if __name__ == "__main__":
    main()
