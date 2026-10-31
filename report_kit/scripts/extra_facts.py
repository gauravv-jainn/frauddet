"""More facts for REPORT_FACTS.md: policy-rule impact, risk-score statistics, latency, dataset columns, drift detail.

Read-only on fraudlens/ (no retraining, no file changes there). Writes only inside report_kit/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parents[1]
FL = KIT.parent / "fraudlens"
ART = FL / "artifacts"
sys.path.insert(0, str(FL))

from fraudkit.config import load_config  # noqa: E402
from fraudkit.data import cached_dataset, clean  # noqa: E402
from fraudkit.decision import policy_flags as model_policy_flags  # noqa: E402
from fraudkit.decision import serving_policy  # noqa: E402
from fraudkit.explain import explain_frame  # noqa: E402
from fraudkit.features import CATEGORICAL, NUMERIC  # noqa: E402
from fraudkit.monitoring import _numeric_shares, drift_report, psi, training_ranges  # noqa: E402
from fraudkit.rules import policy_impact, resolve  # noqa: E402
from fraudkit.score import load_serving_model  # noqa: E402

final = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
prof = json.loads((ART / "reference_profile.json").read_text(encoding="utf-8"))
sc = pd.read_parquet(ART / "test_scored.parquet")
cfg = load_config()
fm = load_serving_model(ART / "model.joblib", bands=cfg.bands.model_dump())
dpol = serving_policy(final, cfg.decision.mode, critical=cfg.bands.critical)
rules_cfg = resolve(cfg.rules.model_dump(), final)
ranges = training_ranges(prof)
out: dict = {}

# --- risk score statistics (0-999) on the test period
score = fm.risk_score(sc["p"].to_numpy())
b = fm.band_bounds(dpol["threshold"])
out["risk_score"] = {
    "genuine_median": float(np.median(score[sc["is_fraud"] == 0])),
    "fraud_median": float(np.median(score[sc["is_fraud"] == 1])),
    "score_at_decision_threshold": int(fm.risk_score(dpol["threshold"])[0]),
    "score_at_high_0.30": int(fm.risk_score(b["high"])[0]),
    "score_at_critical_0.70": int(fm.risk_score(b["critical"])[0]),
    "points_to_double_odds": float(__import__("fraudkit.score", fromlist=["x"]).score_points_to_double_odds()),
    "share_of_test_with_p_below_1e-4": float((sc["p"] < 1e-4).mean()),
    "distinct_p_values_test": int(sc["p"].nunique()),
    "band_bounds": b,
}

# --- policy rules on top of the model (test set, current model)
summ, per_rule = policy_impact(sc, dpol, dpol["review_cost"], rules_cfg, ranges)
summ.to_csv(KIT / "tables" / "policy_rules_impact_test.csv", index=False)
per_rule.to_csv(KIT / "tables" / "policy_rules_per_rule_test.csv", index=False)
print(summ.round(4).to_string())
print(per_rule.round(4).to_string())

# --- decision policy comparison (validation vs test) from metrics.json
d = final["decision"]
rows = []
for name, blk in (("threshold", d["threshold"]), ("expected_loss", d["expected_loss"])):
    for split in ("validation", "test"):
        m = blk[split]
        rows.append({"policy": name, "split": split, **{k: m[k] for k in ("flagged", "tp", "fp", "fn", "precision", "recall", "total_cost", "saved_vs_no_model")}})
pd.DataFrame(rows).to_csv(KIT / "tables" / "decision_policy_comparison.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())

# --- drift detail: December features, and the retired card_txn_count per month
X = sc[NUMERIC + CATEGORICAL]
month = sc["trans_date_trans_time"].dt.to_period("M").astype(str)
dec = drift_report(prof, X[month == "2020-12"], scores=sc.loc[month == "2020-12", "p"].to_numpy())
dec.to_csv(KIT / "tables" / "drift_2020_12_all_features.csv", index=False)
print(dec.head(8).round(3).to_string())

h = pd.read_parquet(ART / "history.parquet", columns=["cc_num", "trans_date_trans_time"])
h = h.sort_values(["cc_num", "trans_date_trans_time"], kind="mergesort")
h["card_txn_count"] = h.groupby("cc_num", sort=False).cumcount()
fit_end = pd.Timestamp(final["split"]["fit"]["end"])
fit_vals = h.loc[h["trans_date_trans_time"] <= fit_end, "card_txn_count"].to_numpy(dtype="float64")
edges = np.unique(np.quantile(fit_vals, np.linspace(0, 1, 11)[1:-1])).tolist()
ref = _numeric_shares(fit_vals, edges)
test_part = h[h["trans_date_trans_time"] > pd.Timestamp(final["split"]["validation"]["end"])]
mm = test_part["trans_date_trans_time"].dt.to_period("M").astype(str)
cc = [{"month": k, "psi_card_txn_count": psi(ref, _numeric_shares(g["card_txn_count"].to_numpy(dtype="float64"), edges))} for k, g in test_part.groupby(mm)]
pd.DataFrame(cc).to_csv(KIT / "tables" / "drift_card_txn_count_by_month.csv", index=False)
print(pd.DataFrame(cc).round(2).to_string())

# --- latency on this machine (warm): one transaction with explanation, and a 10k batch of probabilities
one = sc.iloc[[1000]]
explain_frame(fm, one, policy={"rules": rules_cfg, "ranges": ranges, "decision": dpol, "watch": {}})
t0 = time.perf_counter()
for _ in range(20):
    explain_frame(fm, one, policy={"rules": rules_cfg, "ranges": ranges, "decision": dpol, "watch": {}})
single_ms = (time.perf_counter() - t0) / 20 * 1000
Xb = fm.prepare(sc.iloc[:10000])
t0 = time.perf_counter()
fm.predict_proba(Xb)
batch_ms = (time.perf_counter() - t0) * 1000
out["latency_ms"] = {"explain_one_transaction_warm_mean_of_20": single_ms, "predict_proba_10000_rows": batch_ms}

# --- model file
mf = ART / "model.joblib"
out["model_file"] = {"path": "fraudlens/artifacts/model.joblib", "bytes": mf.stat().st_size, "n_estimators": final["tuning"]["best_params"]["n_estimators"], "num_leaves": final["tuning"]["best_params"]["num_leaves"]}

# --- raw dataset columns, null counts and fraud profile (from the cached Kaggle CSVs; names are dropped by the app's loader)
folder = cached_dataset()
out["raw_dataset_folder"] = str(folder)
if folder:
    tr = pd.read_csv(folder / "fraudTrain.csv")
    te = pd.read_csv(folder / "fraudTest.csv")
    out["raw_rows"] = {"fraudTrain.csv": int(len(tr)), "fraudTest.csv": int(len(te))}
    cols = []
    for c in tr.columns:
        cols.append({"column": c, "pandas_dtype_in_csv": str(tr[c].dtype), "nulls_in_train": int(tr[c].isna().sum()), "distinct_in_train": int(tr[c].nunique()), "example": str(tr[c].iloc[0])[:40] if c not in ("first", "last", "street", "cc_num") else "(not shown: personal data)"})
    pd.DataFrame(cols).to_csv(KIT / "tables" / "dataset_columns_raw.csv", index=False)
    c = clean(tr)
    g = c.groupby("category")["is_fraud"].agg(["count", "sum", "mean"]).rename(columns={"count": "rows", "sum": "frauds", "mean": "fraud_rate"}).sort_values("fraud_rate", ascending=False)
    g.reset_index().to_csv(KIT / "tables" / "fraud_rate_by_category_train.csv", index=False)
    hr = c["trans_date_trans_time"].dt.hour
    night = hr.isin({22, 23, 0, 1, 2, 3})
    out["fraud_rate_train_night_hours_22_to_03"] = float(c.loc[night, "is_fraud"].mean())
    out["fraud_rate_train_other_hours"] = float(c.loc[~night, "is_fraud"].mean())
    out["fraud_amt_train"] = {"max": float(c.loc[c.is_fraud == 1, "amt"].max()), "median": float(c.loc[c.is_fraud == 1, "amt"].median())}
    out["genuine_amt_train"] = {"max": float(c.loc[c.is_fraud == 0, "amt"].max()), "median": float(c.loc[c.is_fraud == 0, "amt"].median())}
    out["amt_over_2000_train"] = {"rows": int((c.amt > 2000).sum()), "frauds": int(c.loc[c.amt > 2000, "is_fraud"].sum())}
    out["columns_dropped_by_loader"] = [x for x in ["first", "last", "street", "zip", "unix_time"] if x in tr.columns]
    print(g.round(4).to_string())

(KIT / "_data" / "extra_facts.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, indent=2, default=str))
sys.stdout.flush()
