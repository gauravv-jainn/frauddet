"""Regenerate report_kit/tables/*.csv and report_kit/_data/*.json from the FraudLens artifacts.

Read-only on fraudlens/: it loads artifacts/, runs the served model for explanations and drift, and writes only
inside report_kit/. Run from the repo root:  fraudlens/.venv/Scripts/python.exe report_kit/scripts/extract_results.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parents[1]
FL = KIT.parent / "fraudlens"
ART = FL / "artifacts"
TABLES, DATA = KIT / "tables", KIT / "_data"
TABLES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
sys.path.insert(0, str(FL))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(df: pd.DataFrame, name: str):
    df.to_csv(TABLES / name, index=False)
    print(f"tables/{name}: {len(df)} rows")


final = load(ART / "metrics.json")
FINAL_VERSION = final["run"]["model_version"]

# ------------------------------------------------------------------ 1. every archived run
rows = []
for d in sorted(glob.glob(str(ART / "runs" / "*"))):
    m = load(Path(d) / "metrics.json")
    r, t, a, dur = m["run"], m["test"], m["test"]["at_threshold"], m["durations_s"]
    ci = t["pr_auc_ci95"]
    rows.append(
        {
            "model_version": r["model_version"],
            "started_utc": r["started_at"],
            "finished_utc": r["finished_at"],
            "wall_clock_seconds": r["total_seconds"],
            "tuning_seconds": dur.get("tuning"),
            "optuna_trials": r["trials"],
            "n_features": len(m["features"]),
            "calibration": m["calibration"].get("method", "isotonic (no comparison stored)"),
            "decision_policy_served": (m.get("decision") or {}).get("served", "threshold (older artifact)"),
            "val_best_cv_pr_auc": m["tuning"]["best_cv_pr_auc"],
            "test_pr_auc": t["pr_auc"],
            "test_pr_auc_ci95_low": ci[0],
            "test_pr_auc_ci95_high": ci[1],
            "test_roc_auc": t["roc_auc"],
            "test_brier_after_calibration": t["brier_after_calibration"],
            "test_brier_before_calibration": t["brier_before_calibration"],
            "threshold": m["threshold"]["value"],
            "flagged": a["flagged"],
            "tp": a["tp"],
            "fp": a["fp"],
            "fn": a["fn"],
            "precision": a["precision"],
            "recall": a["recall"],
            "false_alarms_per_10k_genuine": a["false_alarms_per_10k_genuine"],
            "caught_amount_eur": a["caught_amount"],
            "missed_amount_eur": a["missed_amount"],
            "review_cost_total_eur": a["review_cost_total"],
            "total_cost_eur": a["total_cost"],
            "saved_vs_no_model_eur": a["saved_vs_no_model"],
            "distinct_probabilities": t.get("distinct_probabilities"),
        }
    )
runs = pd.DataFrame(rows)
save(runs, "runs_comparison.csv")

lb = []
for d in sorted(glob.glob(str(ART / "runs" / "*"))):
    m = load(Path(d) / "metrics.json")
    for r in m["leaderboard"]:
        lb.append({"model_version": m["run"]["model_version"], "split": "validation", **r})
save(pd.DataFrame(lb), "leaderboard_by_run.csv")
save(pd.DataFrame(final["leaderboard"]), "leaderboard_final.csv")

# ------------------------------------------------------------------ 2. final-run details
cal = pd.DataFrame(final["calibration"]["comparison"])
save(cal, "calibration_comparison_final.csv")
save(pd.DataFrame(final["decision"]["expected_loss"]["k_grid"]), "decision_expected_loss_k_grid.csv")
save(pd.DataFrame(final["rules_auto"]["spend_spike_grid"]), "rules_spend_spike_grid.csv")
save(
    pd.DataFrame(
        [{"feature": k, "mean_abs_shap_log_odds": v} for k, v in final["global_importance"]["mean_abs_shap"].items()]
    ),
    "global_importance_final.csv",
)
rel = []
for when in ("before", "after"):
    for r in final["test"][f"reliability_{when}"]:
        rel.append({"calibration": when, **r})
save(pd.DataFrame(rel), "reliability_test_final.csv")

sp = final["split"]
save(
    pd.DataFrame(
        [{"part": k, **v} for k, v in sp.items()]
        + [{"part": "raw_train_file (fit+validation)", **final["data"]["train"]}]
    ),
    "dataset_splits.csv",
)

trials = pd.read_csv(ART / "trials.csv")
save(trials, "optuna_trials_final.csv")
tr_sum = {
    "trials": int(len(trials)),
    "best_cv_pr_auc": float(trials["pr_auc_cv"].max()),
    "median_cv_pr_auc": float(trials["pr_auc_cv"].median()),
    "worst_cv_pr_auc": float(trials["pr_auc_cv"].min()),
    "first_trial_cv_pr_auc": float(trials["pr_auc_cv"].iloc[0]),
    "sum_trial_seconds": float(trials["seconds"].sum()),
}

# ------------------------------------------------------------------ 3. feature ablation
ab = load(ART / "feature_ablation.json")
base = next(r["mean"] for r in ab["rows"] if r["candidate"] == "current feature set")
ab_rows = [
    {
        "candidate": r["candidate"],
        "extra_features": ";".join(r["extra_features"]),
        "val_pr_auc_seed42": r["val_pr_auc_by_seed"][0],
        "val_pr_auc_seed7": r["val_pr_auc_by_seed"][1],
        "mean": r["mean"],
        "delta_vs_current_baseline": r["mean"] - base,
        "times_seed_noise": (r["mean"] - base) / ab["seed_noise_baseline"],
    }
    for r in ab["rows"]
]
save(pd.DataFrame(ab_rows), "feature_ablation.csv")

# ------------------------------------------------------------------ 4. drift of the served model on the test period
from fraudkit.features import CATEGORICAL, NUMERIC  # noqa: E402
from fraudkit.monitoring import drift_report  # noqa: E402

prof = load(ART / "reference_profile.json")
sc = pd.read_parquet(ART / "test_scored.parquet")
X = sc[NUMERIC + CATEGORICAL]
overall = drift_report(prof, X, scores=sc["p"].to_numpy())
save(overall, "drift_test_period_overall.csv")
month = sc["trans_date_trans_time"].dt.to_period("M").astype(str)
mrows = []
for mth, idx in sc.groupby(month).groups.items():
    rep = drift_report(prof, X.loc[idx], scores=sc.loc[idx, "p"].to_numpy())
    top = rep.iloc[0]
    mrows.append(
        {
            "month": mth,
            "rows": len(idx),
            "top_feature": top["feature"],
            "top_psi": top["psi"],
            "n_moderate": int((rep["status"] == "moderate").sum()),
            "n_significant": int((rep["status"] == "significant").sum()),
            "score_psi": float(rep.loc[rep["feature"] == "model score", "psi"].iloc[0]),
        }
    )
save(pd.DataFrame(mrows), "drift_test_period_by_month.csv")
# the feature that was removed from the model because it drifted: card_txn_count (context column only)
from fraudkit.monitoring import _numeric_shares, psi  # noqa: E402

h = pd.read_parquet(ART / "history.parquet", columns=["cc_num", "trans_date_trans_time"])
h = h.sort_values(["cc_num", "trans_date_trans_time"], kind="mergesort")
h["card_txn_count"] = h.groupby("cc_num", sort=False).cumcount()
fit_end = pd.Timestamp(final["split"]["fit"]["end"])
fit_vals = h.loc[h["trans_date_trans_time"] <= fit_end, "card_txn_count"].to_numpy(dtype="float64")
test_vals = h.loc[h["trans_date_trans_time"] > pd.Timestamp(final["split"]["validation"]["end"]), "card_txn_count"]
edges = np.unique(np.quantile(fit_vals, np.linspace(0, 1, 11)[1:-1])).tolist()
card_count_psi = psi(_numeric_shares(fit_vals, edges), _numeric_shares(test_vals.to_numpy(dtype="float64"), edges))
print("card_txn_count PSI test vs fit:", card_count_psi)

# ------------------------------------------------------------------ 5. three real SHAP explanations
from fraudkit.config import load_config  # noqa: E402
from fraudkit.decision import serving_policy  # noqa: E402
from fraudkit.explain import explain_frame  # noqa: E402
from fraudkit.monitoring import training_ranges  # noqa: E402
from fraudkit.rules import resolve  # noqa: E402
from fraudkit.score import load_serving_model  # noqa: E402

cfg = load_config()
fm = load_serving_model(ART / "model.joblib", bands=cfg.bands.model_dump())
dpol = serving_policy(final, cfg.decision.mode, critical=cfg.bands.critical)
pol = {
    "rules": resolve(cfg.rules.model_dump(), final),
    "ranges": training_ranges(prof),
    "decision": dpol,
    "watch": {},
}
thr = final["threshold"]["value"]
tp = sc[(sc["is_fraud"] == 1) & (sc["p"] >= thr)].nlargest(1, "p")
fp = sc[(sc["is_fraud"] == 0) & (sc["p"] >= thr)].nlargest(1, "p")
fn = sc[(sc["is_fraud"] == 1) & (sc["p"] < thr)].nlargest(1, "p")
picks = [("caught fraud (true positive, highest p)", tp), ("false alarm (genuine flagged, highest p)", fp),
         ("missed fraud (false negative, highest p below threshold)", fn)]
ex_out = []
for label, df in picks:
    e = explain_frame(fm, df, top_raising=4, top_lowering=2, policy=pol)[0]
    row = df.iloc[0]
    ex_out.append(
        {
            "case": label,
            "trans_num": str(row["trans_num"]),
            "row_label_is_fraud": int(row["is_fraud"]),
            "amount": float(row["amt"]),
            "category": str(row["category"]),
            "hour": int(row["hour"]),
            "stored_p": float(row["p"]),
            "recomputed_p": e.probability,
            "risk_score": e.risk_score,
            "model_band": e.model_band,
            "final_band": e.band,
            "decision": e.decision,
            "expected_loss": e.expected_loss,
            "summary": e.summary,
            "raising": [{"text": r.text, "shap": r.contribution} for r in e.raising],
            "lowering": [{"text": r.text, "shap": r.contribution} for r in e.lowering],
            "rules": [{"rule": r.rule, "band": r.band, "text": r.text} for r in e.rules],
            "base_value_log_odds": e.base_value,
        }
    )
(DATA / "shap_examples.json").write_text(json.dumps(ex_out, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------------ 6. auth roles, git timeline, test inventory
from fraudkit.auth import PERMISSIONS, ROLES  # noqa: E402

save(
    pd.DataFrame(
        [{"permission": p, **{r: (r in roles) for r in ROLES}} for p, roles in PERMISSIONS.items()]
    ),
    "role_permission_matrix.csv",
)

import subprocess  # noqa: E402

log = subprocess.run(  # noqa: S603
    ["git", "log", "--reverse", "--date=short", "--pretty=format:@@%h|%ad|%an|%s", "--shortstat"],  # noqa: S607
    cwd=KIT.parent, capture_output=True, text=True, encoding="utf-8", check=True,
).stdout
commits, cur = [], None
for line in log.splitlines():
    if line.startswith("@@"):
        h_, d_, a_, s_ = line[2:].split("|", 3)
        cur = {"hash": h_, "date": d_, "author": a_, "subject": s_, "files_changed": 0, "insertions": 0, "deletions": 0}
        commits.append(cur)
    elif "changed" in line and cur is not None:
        import re

        for key, pat in (("files_changed", r"(\d+) file"), ("insertions", r"(\d+) insertion"), ("deletions", r"(\d+) deletion")):
            mm = re.search(pat, line)
            cur[key] = int(mm.group(1)) if mm else 0
save(pd.DataFrame(commits), "git_timeline.csv")

summary = {
    "final_version": FINAL_VERSION,
    "trials": tr_sum,
    "ablation_seed_noise": ab["seed_noise_baseline"],
    "ablation_model_version": ab["model_version"],
    "ablation_kept": ab["kept"],
    "card_txn_count_psi_test_vs_fit": card_count_psi,
    "threshold": thr,
    "decision_policy_served": dpol,
}
(DATA / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2)[:1500])
os._exit(0)
