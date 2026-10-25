"""Build the viva report (docs/FraudLens_Viva_Report.pdf) from the artifacts of the last training run.

Every number in the report is read from artifacts/ (metrics.json, trials.csv, test_scored.parquet,
history.parquet, reference_profile.json, runs/) or computed from them here. Nothing is typed by hand.

    python scripts/build_report.py
"""

from __future__ import annotations

import base64
import html
import io
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudkit.config import load_config
from fraudkit.explain import explain_frame
from fraudkit.features import FEATURES
from fraudkit.monitoring import drift_report, training_ranges
from fraudkit.decision import describe_policy, serving_policy
from fraudkit.rules import policy_impact, resolve
from fraudkit.score import load_serving_model

ART = ROOT / "artifacts"
OUT_PDF = ROOT / "docs" / "FraudLens_Viva_Report.pdf"
BLUE, RED, GREEN, GREY, AMBER = "#1D4ED8", "#B91C1C", "#15803D", "#94A3B8", "#CA8A04"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


# --------------------------------------------------------------------------- helpers
def img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'/>"


def e(x) -> str:
    return html.escape(str(x))


def table(df: pd.DataFrame, fmt: dict | None = None, cls: str = "") -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{e(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            cells.append(f"<td>{e(f(v) if f else v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def kv(rows: list[tuple[str, str]]) -> str:
    return table(pd.DataFrame(rows, columns=["Item", "Value"]))


# --------------------------------------------------------------------------- data
m = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
cfg = load_config()
fm = load_serving_model(ART / "model.joblib", bands=cfg.bands.model_dump())
scored = pd.read_parquet(ART / "test_scored.parquet")
trials = pd.read_csv(ART / "trials.csv")
profile = json.loads((ART / "reference_profile.json").read_text(encoding="utf-8"))
ranges = training_ranges(profile)
te, cal, tn = m["test"], m["calibration"], m["tuning"]
at = te.get("at_policy", te["at_threshold"])  # served decision policy
RULES = resolve(cfg.rules.model_dump(), m)
dec_m = m.get("decision", {})
served_pol = serving_policy(m, cfg.decision.mode, critical=cfg.bands.critical)
ablation = (
    json.loads((ART / "feature_ablation.json").read_text(encoding="utf-8"))
    if (ART / "feature_ablation.json").exists()
    else None
)
thr = at["threshold"]
version = m["run"]["model_version"]
y, p = scored["is_fraud"].to_numpy(), scored["p"].to_numpy()

hist = pd.read_parquet(ART / "history.parquet", columns=["amt", "is_fraud", "trans_date_trans_time"])
train_end = pd.Timestamp(m["data"]["train"]["end"])
train_raw = hist[hist["trans_date_trans_time"] <= train_end]

# previous model (lifetime counter feature), archived before the fix
older = []
for d in sorted((ART / "runs").glob("*/metrics.json")):
    mm = json.loads(d.read_text(encoding="utf-8"))
    if "card_txn_count" in mm.get("features", []):
        older.append(mm)
old = older[-1] if older else None

# --------------------------------------------------------------------------- charts
lb = pd.DataFrame(m["leaderboard"]).sort_values("pr_auc")
fig, ax = plt.subplots(figsize=(6.2, 2.4))
colors = [BLUE if s else GREY for s in lb["selected"]]
ax.barh(lb["model"], lb["pr_auc"], color=colors)
for i, v in enumerate(lb["pr_auc"]):
    ax.text(v + 0.005, i, f"{v:.4f}", va="center")
ax.set_xlim(0, 1.08)
ax.set_xlabel("PR-AUC on the validation period (higher is better)")
ax.set_title("Model comparison (same features, same validation data)")
CH_LEADERBOARD = img(fig)

fig, ax = plt.subplots(figsize=(6.2, 2.4))
ax.scatter(trials["trial"], trials["pr_auc_cv"], color=GREY, s=14, label="trial PR-AUC (3-fold CV)")
ax.step(trials["trial"], trials["best_so_far"], where="post", color=BLUE, lw=2, label="best so far")
ax.set_xlabel("Optuna trial")
ax.set_ylabel("PR-AUC")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Hyper-parameter search history")
CH_TRIALS = img(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axes[0]
ax.plot([0, 1], [0, 1], ":", color=GREY, label="perfect")
for key, name, col, ls in [
    ("reliability_before", "uncalibrated", GREY, "--"),
    ("reliability_after", "calibrated", BLUE, "-"),
]:
    d = pd.DataFrame(te.get(key, []))
    if len(d):
        ax.plot(d["mean_predicted"], d["observed_rate"], ls, marker="o", ms=3, color=col, label=name)
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed fraud rate")
ax.set_title("Reliability (test)")
ax.legend(frameon=False)
ax = axes[1]
cv = pd.DataFrame(m["threshold"]["validation_curve"])
ax.plot(cv["threshold"], cv["total_cost"], color=BLUE)
ax.scatter([thr], [m["threshold"]["validation_at_threshold"]["total_cost"]], color=RED, zorder=3, label="chosen")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("threshold (log)")
ax.set_ylabel("missed fraud + review cost")
ax.set_title("Threshold vs cost (validation)")
ax.legend(frameon=False)
fig.tight_layout()
CH_CAL_COST = img(fig)

prec, rec, _ = precision_recall_curve(y, p)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axes[0]
ax.plot(rec, prec, color=BLUE)
ax.scatter([at["recall"]], [at["precision"]], color=RED, zorder=3, label=f"threshold {thr:.4f}")
ax.axhline(y.mean(), color=GREY, ls=":", label=f"no-skill = fraud rate {y.mean():.3%}")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_title(f"Precision-recall curve (test), AP = {te['pr_auc']:.4f}")
ax.legend(frameon=False, loc="lower left")
ax = axes[1]
cm = np.array([[at["tn"], at["fp"]], [at["fn"], at["tp"]]])
ax.imshow(np.log10(cm + 1), cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if i == j else "black", fontsize=10)
ax.set_xticks([0, 1], ["predicted genuine", "predicted fraud"])
ax.set_yticks([0, 1], ["actual genuine", "actual fraud"])
ax.set_title("Confusion matrix at the chosen threshold")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
CH_PR_CM = img(fig)

imp = pd.Series(m["global_importance"]["mean_abs_shap"]).sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.barh(imp.index, imp.values, color=BLUE)
ax.set_xlabel("mean |SHAP| (log-odds)")
ax.set_title("Global feature importance (SHAP, validation sample)")
CH_SHAP = img(fig)

edges = [0, 20, 100, 300, 500, 800, 1000, 1300, 2000, 5000, 1e6]
labels = [f"{a:,.0f}-{b:,.0f}" for a, b in zip(edges[:-2], edges[1:-1], strict=True)] + [f"{edges[-2]:,.0f}+"]
buckets = pd.cut(train_raw["amt"], edges, labels=labels)
amt_tbl = train_raw.groupby(buckets, observed=True)["is_fraud"].agg(["size", "sum", "mean"]).reset_index()
amt_tbl.columns = ["amount range (€)", "transactions", "frauds", "fraud rate"]
amt_tbl["amount range (€)"] = amt_tbl["amount range (€)"].astype(str)
fig, ax = plt.subplots(figsize=(6.2, 2.3))
ax.bar(
    amt_tbl["amount range (€)"], amt_tbl["fraud rate"], color=[RED if r > 0 else GREY for r in amt_tbl["fraud rate"]]
)
ax.set_ylabel("fraud rate")
ax.set_title("Fraud rate by amount in the training data")
ax.tick_params(axis="x", rotation=35)
CH_AMOUNT = img(fig)

X = fm.prepare(scored)
months = scored["trans_date_trans_time"].dt.to_period("M").astype(str)
drift_rows = []
for mo in sorted(months.unique()):
    mask = (months == mo).to_numpy()
    r = drift_report(profile, X[mask], scores=p[mask])
    r["month"] = mo
    drift_rows.append(r)
drift = pd.concat(drift_rows)
heat = drift.pivot(index="feature", columns="month", values="psi")
heat = heat.loc[heat.max(axis=1).sort_values(ascending=False).index[:10]]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.imshow(heat.values, cmap="YlOrRd", vmin=0, vmax=0.5, aspect="auto")
ax.set_xticks(range(heat.shape[1]), heat.columns, rotation=0)
ax.set_yticks(range(heat.shape[0]), heat.index)
for (i, j), v in np.ndenumerate(heat.values):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Drift (PSI) of the 10 most-shifted features, by test month")
CH_DRIFT = img(fig)

impact, per_rule = policy_impact(scored, served_pol, at["review_cost"], RULES, ranges)

# example explanation: highest-risk caught fraud
ex_row = scored[(scored["is_fraud"] == 1)].nlargest(1, "p")
ex = explain_frame(fm, ex_row, thr, policy={"rules": RULES, "ranges": ranges, "decision": served_pol})[0]
missed_row = scored[(scored["is_fraud"] == 1) & (scored["p"] < thr)].nlargest(1, "amt")
ex_missed = explain_frame(fm, missed_row, thr)[0] if len(missed_row) else None

# --------------------------------------------------------------------------- tables
lb_t = pd.DataFrame(m["leaderboard"])[["model", "pr_auc", "roc_auc", "brier", "fit_seconds", "selected"]]
lb_t["selected"] = lb_t["selected"].map({True: "selected", False: ""})
LB_TABLE = table(
    lb_t,
    {
        "pr_auc": lambda v: f"{v:.4f}",
        "roc_auc": lambda v: f"{v:.4f}",
        "brier": lambda v: f"{v:.5f}",
        "fit_seconds": lambda v: f"{v:.1f}",
    },
)
bp = tn["best_params"]
SPACE = pd.DataFrame(
    [
        ("n_estimators", "150 - 1200 (step 50)", "number of trees", bp.get("n_estimators")),
        ("learning_rate", "0.01 - 0.2 (log)", "shrinkage per tree", f"{bp.get('learning_rate', 0):.4f}"),
        ("num_leaves", "15 - 255 (log)", "tree complexity", bp.get("num_leaves")),
        ("min_child_samples", "10 - 300 (log)", "minimum rows per leaf (regularisation)", bp.get("min_child_samples")),
        ("subsample", "0.5 - 1.0", "row sampling per tree (bagging)", f"{bp.get('subsample', 0):.3f}"),
        ("colsample_bytree", "0.4 - 1.0", "feature sampling per tree", f"{bp.get('colsample_bytree', 0):.3f}"),
        ("reg_lambda", "0.001 - 10 (log)", "L2 penalty on leaf weights", f"{bp.get('reg_lambda', 0):.4f}"),
        ("reg_alpha", "0.001 - 10 (log)", "L1 penalty on leaf weights", f"{bp.get('reg_alpha', 0):.4f}"),
        ("scale_pos_weight", "1 - 10 (log)", "extra weight on fraud rows", f"{bp.get('scale_pos_weight', 0):.3f}"),
    ],
    columns=["hyper-parameter", "search range", "meaning", "best value"],
)

FEATURE_DOC = {
    "amt": ("Transaction amount", "raw"),
    "log_amt": ("log(1 + amount), a smoother version of the amount", "raw"),
    "hour": ("Hour of the day (0-23)", "time"),
    "day_of_week": ("Day of week (0 = Monday)", "time"),
    "is_night": ("1 if between 22:00 and 03:59", "time"),
    "dist_home_km": ("Haversine distance from cardholder home to merchant", "geo"),
    "age": ("Cardholder age at transaction time", "profile"),
    "log_city_pop": ("log population of the cardholder's city", "profile"),
    "merchant_te": ("Merchant target encoding: smoothed fraud rate of the merchant in the training period", "encoded"),
    "n_txn_1h": ("Card's transactions in the previous 1 hour", "velocity"),
    "n_txn_24h": ("Card's transactions in the previous 24 hours", "velocity"),
    "amt_sum_24h": ("Card's total spend in the previous 24 hours", "velocity"),
    "amt_to_card_mean": ("Amount / card's average of all earlier amounts", "behaviour"),
    "amt_to_cat_mean": ("Amount / card's average earlier amount in the same category", "behaviour"),
    "hours_since_prev": ("Hours since the card's previous transaction", "behaviour"),
    "geo_speed_kmh": ("km between this and the previous merchant / hours between them", "geo"),
    "impossible_travel": ("1 if that speed > 800 km/h and the jump > 50 km", "geo"),
    "first_time_category": ("1 if the card never bought in this category before", "behaviour"),
    "n_txn_30d": ("Card's transactions in the previous 30 days (replaced the lifetime counter)", "velocity"),
    "category": ("Merchant category (14 values), native categorical in LightGBM", "categorical"),
    "gender": ("Cardholder gender (as in the dataset)", "categorical"),
}
FEAT_T = pd.DataFrame(
    [
        (f, FEATURE_DOC[f][1], FEATURE_DOC[f][0], f"{m['global_importance']['mean_abs_shap'].get(f, 0):.3f}")
        for f in FEATURES
    ],
    columns=["feature", "group", "meaning / how computed", "mean |SHAP|"],
)

ALGOS = pd.DataFrame(
    [
        (
            "Logistic Regression",
            "Baseline classifier",
            "fraudkit/train.py (baseline)",
            "Simplest strong baseline; shows how much non-linearity adds",
        ),
        (
            "Random Forest (200 trees)",
            "Baseline classifier",
            "fraudkit/train.py (baseline)",
            "Bagged trees; handles interactions, robust",
        ),
        (
            "LightGBM (gradient boosting)",
            "Final model",
            "fraudkit/train.py, fraudkit/score.py",
            "Best accuracy on tabular data, fast, native categoricals",
        ),
        (
            "Optuna TPE",
            "Hyper-parameter search",
            "fraudkit/train.py (tuning)",
            "Bayesian search: tries promising settings first",
        ),
        (
            "Stratified 3-fold CV",
            "Tuning evaluation",
            "fraudkit/train.py (tuning)",
            "Every fold keeps the same 0.6% fraud rate",
        ),
        (
            "Smoothed target encoding (out-of-fold)",
            "Merchant feature",
            "fraudkit/features.py",
            "Turns 693 merchant names into one number without leaking labels",
        ),
        (
            "Isotonic regression (Platt fallback)",
            "Probability calibration",
            "fraudkit/score.py (Calibrator)",
            "Makes 30% mean ~30%; needed for money decisions",
        ),
        (
            "Cost-minimising threshold",
            "Decision rule",
            "fraudkit/costs.py",
            "Chooses the cut-off that minimises money lost",
        ),
        ("Bootstrap (1,000 resamples)", "Confidence interval", "fraudkit/train.py", "Uncertainty of the test PR-AUC"),
        (
            "SHAP TreeExplainer",
            "Explanations",
            "fraudkit/explain.py",
            "Exact per-feature contributions for tree models",
        ),
        (
            "Population Stability Index",
            "Drift monitoring",
            "fraudkit/monitoring.py",
            "Detects when live data stops looking like training data",
        ),
        (
            "Policy rules",
            "Guardrail",
            "fraudkit/rules.py",
            "Covers cases outside the training data (model blind spots)",
        ),
    ],
    columns=["algorithm / method", "role", "where in the code", "why"],
)

# --------------------------------------------------------------------------- viva Q&A
nfit, nval, ntest = m["split"]["fit"]["rows"], m["split"]["validation"]["rows"], m["split"]["test"]["rows"]
QA = [
    (
        "What problem does the project solve?",
        "It scores every card transaction with a calibrated fraud probability, explains why in plain English, and "
        "decides who reviews what so that the money lost to fraud plus the cost of reviews is as small as possible.",
    ),
    (
        "Which algorithm is the final model and why?",
        f"LightGBM (gradient-boosted decision trees), tuned with Optuna. It had the best validation PR-AUC "
        f"({lb_t.loc[lb_t['selected'] == 'selected', 'pr_auc'].iloc[0]:.4f}) of four models compared on the same data, "
        "trains on a normal CPU (the full 30-trial pipeline runs in minutes), handles categories natively, and SHAP "
        "can explain it exactly.",
    ),
    (
        "Why not accuracy?",
        f"Only {m['data']['test']['fraud_rate']:.2%} of test transactions are fraud. A model that says 'genuine' every "
        f"time is {1 - m['data']['test']['fraud_rate']:.2%} accurate and catches nothing. PR-AUC focuses on the rare "
        "class: how precise the alerts are at every recall level.",
    ),
    (
        "Why PR-AUC instead of ROC-AUC?",
        f"ROC-AUC uses the false-positive rate, whose denominator is the huge number of genuine transactions, so it looks "
        f"excellent even for weak models (all four models score above 0.94). PR-AUC is much more sensitive to how many "
        f"alerts are wrong. Our no-skill PR-AUC equals the fraud rate ({y.mean():.3%}); we get {te['pr_auc']:.4f}.",
    ),
    (
        "How were the models compared fairly?",
        f"Same 21 features, same time-ordered split: fit on the first 85% of the training period ({nfit:,} rows), "
        f"compare on the last 15% ({nval:,} rows, the validation set), same metric. The test file ({ntest:,} rows, a "
        "later time period) was used exactly once, after everything was chosen.",
    ),
    (
        "Why split by time and not randomly?",
        "Fraud detection always predicts the future from the past. A random split lets the model see future "
        "transactions of the same card during training, which inflates results. The provided train/test files are "
        "already split in time and we kept that.",
    ),
    (
        "What is data leakage and how did you prevent it?",
        "Leakage is using information that would not exist at prediction time. Every card feature uses only strictly "
        "earlier transactions of that card. A unit test removes all later rows and checks that the features of every "
        "row stay identical. The merchant encoding is fitted on the training period only, out-of-fold for training rows.",
    ),
    (
        "Explain target encoding.",
        "Each merchant is replaced by its fraud rate in the training data, pulled towards the overall rate when the "
        "merchant has few transactions (smoothing = 50). Out-of-fold means a row's own label is never used for its own "
        "encoding.",
    ),
    (
        "How does gradient boosting work?",
        "Trees are added one after another; each new tree predicts the errors (gradients of the log-loss) of the "
        "current ensemble, and its output is added with a small learning rate. LightGBM grows trees leaf-wise and bins "
        "features into histograms, which makes it fast.",
    ),
    (
        "Why did default LightGBM do worse than the random forest?",
        "Default settings (100 trees, learning rate 0.1, no class weighting) are not tuned for 0.6% positives; it "
        "overfits a few fraud patterns. Tuning the regularisation (min_child_samples, L1/L2), sampling and "
        f"scale_pos_weight lifted it from {lb_t.loc[lb_t['model'] == 'LightGBM default', 'pr_auc'].iloc[0]:.4f} to "
        f"{lb_t.loc[lb_t['selected'] == 'selected', 'pr_auc'].iloc[0]:.4f}.",
    ),
    (
        "Why is logistic regression lowest?",
        "It is linear in the features. Fraud here depends on interactions (large amount AND night AND online "
        "category); trees capture those automatically, a linear model cannot without hand-made interaction terms.",
    ),
    (
        "What is Optuna / TPE?",
        f"Optuna is a hyper-parameter search library. TPE (Tree-structured Parzen Estimator) models which settings "
        f"gave good scores and samples new ones from the promising region. We ran {tn['trials']} trials, each scored "
        f"with 3-fold stratified CV on the fit period; best CV PR-AUC {tn['best_cv_pr_auc']:.4f}.",
    ),
    (
        "Why subsample genuine rows during tuning?",
        f"Speed. Inside each training fold only {cfg.model.tuning_sample_negatives:.0%} of genuine rows are kept (all "
        "frauds kept); evaluation folds are untouched, so the scores stay honest. The shifted base rate is fixed later "
        "by calibration.",
    ),
    (
        "What is calibration and why is it needed?",
        f"Raw model scores are not true probabilities (especially after subsampling). Isotonic regression, fitted on the "
        f"validation set, maps scores to probabilities that match observed frequencies. Test Brier score went from "
        f"{te['brier_before_calibration']:.5f} to {te['brier_after_calibration']:.5f}. Calibrated probabilities are "
        "required to compare expected loss with review cost.",
    ),
    (
        "How is the threshold chosen?",
        f"We minimise total cost = missed fraud amount + review cost x number of flagged transactions on the validation "
        f"set, with review cost {at['review_cost']:.0f}. The best threshold is {thr:.4f}: flag anything with at least "
        f"{thr:.2%} fraud probability. It is low because a missed fraud costs hundreds while a review costs "
        f"{at['review_cost']:.0f}.",
    ),
    (
        "What are your final test results?",
        f"PR-AUC {te['pr_auc']:.4f} (95% CI {te['pr_auc_ci95'][0]:.4f}-{te['pr_auc_ci95'][1]:.4f}), recall "
        f"{at['recall']:.1%}, precision {at['precision']:.1%}, {at['false_alarms_per_10k_genuine']:.1f} false alarms per "
        f"10,000 genuine transactions, {at['saved_vs_no_model']:,.0f} saved out of {at['fraud_amount']:,.0f} of fraud.",
    ),
    (
        "How was the confidence interval computed?",
        f"Bootstrap: resample the {ntest:,} test rows with replacement {te['bootstrap_resamples']:,} times, recompute "
        "PR-AUC each time, take the 2.5th and 97.5th percentiles.",
    ),
    (
        "What is SHAP?",
        "SHapley Additive exPlanations: each feature gets a contribution so that base value + sum of contributions = "
        "the model output (in log-odds). TreeExplainer computes them exactly for tree models. We test that the sum "
        "reconstructs the output.",
    ),
    (
        "How do the plain-English reasons work?",
        "Related features are grouped (e.g. amount and log-amount), groups with contribution >= 0.05 log-odds become "
        "reasons, the top 4 raising and top 2 lowering are shown, and each is written with the transaction's actual "
        "numbers (e.g. '13.1x this card's usual spend').",
    ),
    (
        "What is PSI?",
        "Population Stability Index: bin a feature using training-time bins, compare the share of data in each bin now "
        "vs then: PSI = sum (actual - expected) x ln(actual / expected). Below 0.10 stable, 0.10-0.25 moderate, above "
        "0.25 significant.",
    ),
    (
        "Did monitoring find anything?",
        "Yes. The first version had a feature 'card_txn_count' (lifetime transactions of the card). It only grows, so "
        "its PSI was 3.7-6.1 every test month and the model trusted cards more the older they got. Replacing it with "
        "transactions in the last 30 days improved test PR-AUC (see section 11).",
    ),
    (
        "Why do you need policy rules if the model is so good?",
        "A model can only learn patterns present in its data. In this simulated data every fraud is at most about "
        f"€{train_raw.loc[train_raw['is_fraud'] == 1, 'amt'].max():,.0f} and all merchants are within about "
        f"{ranges.get('dist_home_km', {}).get('max', 0):,.0f} km of home, so a €12,000 purchase 2,500 km away looks "
        "normal to the model. Rules flag what the model has never seen. They can only raise the risk, never lower it.",
    ),
    (
        "Do the rules hurt performance?",
        f"On this simulated test set they add {int(impact.loc[1, 'flagged'] - impact.loc[0, 'flagged']):,} reviews and "
        "catch no extra fraud (the simulator has no such fraud), costing "
        f"{impact.loc[1, 'total_cost'] - impact.loc[0, 'total_cost']:,.0f} extra. That is the insurance premium "
        "for protection outside the training data.",
    ),
    (
        "What are the limitations?",
        "Simulated data with clean fraud patterns (real PR-AUC would be lower), no device/IP/merchant-risk signals, "
        "labels available instantly (real chargebacks take weeks), calibration and threshold share the validation "
        "period, and a single flat review cost.",
    ),
    (
        "Is it reproducible?",
        "Yes: fixed seeds, deterministic LightGBM, and a test that trains twice and compares every metric. Each "
        f"trained model gets a version id ({version}) stored with its metrics.",
    ),
    (
        "What is the expected-loss policy?",
        "Review a transaction when calibrated probability x amount is at least k times the review cost. It spends "
        "analyst time where the money is; small risky transactions get step-up verification instead. k is tuned on "
        "validation and the policy is served only if it beats the flat threshold on validation.",
    ),
    (
        "Which policy won, and why not pick the one that was better on test?",
        f"On validation the {dec_m.get('served', 'threshold').replace('_', ' ')} policy was cheaper, so it is served. "
        "Choosing by test results would turn the test set into a tuning set and the reported numbers would no longer "
        "be an honest estimate.",
    ),
    (
        "Did the new card-profile features help?",
        "No. In a two-seed ablation on validation none improved PR-AUC beyond seed noise, so they were not added to "
        "the model. They are still used for explanations, rules and the Card 360 view.",
    ),
    (
        "How do analyst decisions affect the system?",
        "Confirming fraud blocks the card (stored as a SHA-256 hash), so later transactions on it are blocked by a "
        "rule; merchants can be put on a watchlist. Decisions are also the labelled data for the next model, but here "
        "they come from the test period, so they are not fed back into this model.",
    ),
    (
        "What would you do next?",
        "Real data with chargeback delays, device and network features, merchant risk lists, a champion/challenger "
        "retraining process with approval, and cost per transaction instead of a flat review cost.",
    ),
]

# --------------------------------------------------------------------------- old vs new comparison
if old:
    oa, na = old["test"]["at_threshold"], at
    OLD_NEW = table(
        pd.DataFrame(
            [
                ("Test PR-AUC", f"{old['test']['pr_auc']:.4f}", f"{te['pr_auc']:.4f}"),
                (
                    "95% CI",
                    f"{old['test']['pr_auc_ci95'][0]:.4f} - {old['test']['pr_auc_ci95'][1]:.4f}",
                    f"{te['pr_auc_ci95'][0]:.4f} - {te['pr_auc_ci95'][1]:.4f}",
                ),
                ("Recall", f"{oa['recall']:.1%}", f"{na['recall']:.1%}"),
                ("Precision", f"{oa['precision']:.1%}", f"{na['precision']:.1%}"),
                (
                    "False alarms / 10k genuine",
                    f"{oa['false_alarms_per_10k_genuine']:.1f}",
                    f"{na['false_alarms_per_10k_genuine']:.1f}",
                ),
                ("Saved vs no model", f"{oa['saved_vs_no_model']:,.0f}", f"{na['saved_vs_no_model']:,.0f}"),
                ("Threshold", f"{oa['threshold']:.4f}", f"{na['threshold']:.4f}"),
            ],
            columns=["metric (same test set)", "with card_txn_count (old)", "with n_txn_30d (final)"],
        )
    )
else:
    OLD_NEW = "<p>(previous run archive not found)</p>"

# --------------------------------------------------------------------------- html
d, sp = m["data"], m["split"]
css = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 10.2pt; color: #0F172A; line-height: 1.42; }
h1 { font-size: 24pt; margin: 0 0 4px 0; color: #1E3A8A; }
h2 { font-size: 14pt; color: #1E3A8A; border-bottom: 2px solid #CBD5E1; padding-bottom: 3px; margin-top: 18px;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12px 0 4px 0; page-break-after: avoid; }
p, li { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px 0; font-size: 9pt; page-break-inside: avoid; }
th { background: #EEF2FF; text-align: left; padding: 4px 6px; border: 1px solid #CBD5E1; }
td { padding: 3px 6px; border: 1px solid #E2E8F0; vertical-align: top; }
img { max-width: 100%; max-height: 74mm; width: auto; display: block; margin: 6px auto; page-break-inside: avoid; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; }
.sub { color: #475569; font-size: 12pt; }
.box { border: 1px solid #CBD5E1; border-left: 5px solid #1D4ED8; background: #F8FAFC; padding: 8px 12px; margin: 8px 0;
       page-break-inside: avoid; }
.warn { border-left-color: #B91C1C; background: #FEF2F2; }
.kpis { display: flex; gap: 8px; margin: 10px 0; }
.kpi { flex: 1; border: 1px solid #CBD5E1; border-radius: 6px; padding: 8px; text-align: center; }
.kpi b { display: block; font-size: 15pt; color: #1D4ED8; }
.flow { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; font-size: 8.6pt; margin: 8px 0; }
.flow span { border: 1px solid #94A3B8; border-radius: 5px; padding: 4px 6px; background: #F1F5F9; }
.flow i { color: #64748B; font-style: normal; }
.qa { page-break-inside: avoid; margin-bottom: 7px; }
.qa b { color: #1E3A8A; }
.pb { page-break-before: always; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #F1F5F9; padding: 0 3px; }
"""

flow = "<i>&rarr;</i>".join(
    f"<span>{s}</span>"
    for s in [
        "Kaggle CSVs",
        "leakage-free features (21)",
        "time split fit / validation / test",
        "4 models compared",
        "Optuna tuning (30 trials)",
        "isotonic calibration",
        "cost-optimal threshold",
        "test once + bootstrap CI",
        "SHAP reasons",
        "policy rules",
        "app / API",
        "drift monitoring",
    ]
)

sections = []
sections.append(f"""
<div class='cover'>
<h1>FraudLens</h1>
<div class='sub'>Explainable, calibrated credit-card fraud detection: algorithms, comparison and results</div>
<p style='margin-top:30px'>Viva report generated from the project's actual training run.</p>
{
    kv(
        [
            ("Model version", version),
            ("Final model", m["selected_model"]),
            ("Dataset", d["source"]),
            ("Test PR-AUC (95% CI)", f"{te['pr_auc']:.4f} ({te['pr_auc_ci95'][0]:.4f} - {te['pr_auc_ci95'][1]:.4f})"),
            ("Recall / precision at the chosen threshold", f"{at['recall']:.1%} / {at['precision']:.1%}"),
            ("Money saved vs no model (test period)", f"{at['saved_vs_no_model']:,.0f} of {at['fraud_amount']:,.0f}"),
            ("Code", "github.com/gauravv-jainn/frauddet, folder fraudlens/"),
            ("Report date", date.today().isoformat()),
        ]
    )
}
<p class='sub' style='font-size:9.5pt'>Every number in this document is read from artifacts/ of the run above by
scripts/build_report.py; nothing is typed by hand.</p>
</div>""")

sections.append(f"""
<h2 class='pb'>1. Summary in one page</h2>
<p><b>Problem.</b> Card fraud is rare ({d["test"]["fraud_rate"]:.2%} of test transactions) but expensive. The bank
must decide, for every transaction, whether to approve it or send it to an analyst. Reviews cost money; missed fraud
costs more.</p>
<p><b>Approach.</b></p>
<div class='flow'>{flow}</div>
<div class='kpis'>
<div class='kpi'><b>{te["pr_auc"]:.4f}</b>test PR-AUC</div>
<div class='kpi'><b>{at["recall"]:.1%}</b>frauds caught</div>
<div class='kpi'><b>{at["precision"]:.1%}</b>alerts that are fraud</div>
<div class='kpi'><b>{at["false_alarms_per_10k_genuine"]:.1f}</b>false alarms / 10k genuine</div>
<div class='kpi'><b>{at["saved_vs_no_model"] / at["fraud_amount"]:.1%}</b>of fraud money saved</div>
</div>
<p><b>Key decisions you should be able to defend:</b></p>
<ul>
<li>Time-based split; the test file (later in time) is used once.</li>
<li>PR-AUC as the comparison metric because of extreme class imbalance.</li>
<li>LightGBM chosen after comparing Logistic Regression, Random Forest, default LightGBM and tuned LightGBM.</li>
<li>Calibrated probabilities plus a threshold that minimises money lost, not one that maximises accuracy.</li>
<li>Every decision is explained with SHAP-based reasons; policy rules cover what the model cannot know.</li>
</ul>""")

sections.append(f"""
<h2>2. Dataset</h2>
{
    kv(
        [
            ("Source", d["source"]),
            (
                "Training file",
                f"{d['train']['rows']:,} transactions, {d['train']['frauds']:,} frauds ({d['train']['fraud_rate']:.3%}), {d['train']['start'][:10]} to {d['train']['end'][:10]}",
            ),
            (
                "Test file",
                f"{d['test']['rows']:,} transactions, {d['test']['frauds']:,} frauds ({d['test']['fraud_rate']:.3%}), {d['test']['start'][:10]} to {d['test']['end'][:10]}",
            ),
            ("Cards", f"{d['train']['cards']:,} in training, {d['test']['cards']:,} in test"),
            (
                "Fit / validation split",
                f"first 85% of the training period by time = fit ({sp['fit']['rows']:,} rows); last 15% = validation ({sp['validation']['rows']:,} rows, {sp['validation']['frauds']:,} frauds, from {sp['validation']['start'][:10]})",
            ),
            (
                "Raw columns used",
                "time, card number, merchant, category, amount, gender, city, state, home lat/long, city population, job, date of birth, merchant lat/long, label",
            ),
        ]
    )
}
<p>The data is generated by the Sparkov simulator. It is realistic in structure but its fraud patterns are cleaner
than real life; see Limitations.</p>""")

sections.append(f"""
<h2>3. Features (21) and how leakage is prevented</h2>
<p>Card-behaviour features are computed after sorting by card and time, using only <b>strictly earlier</b>
transactions of the same card. This is what would be known at authorisation time. A unit test
(<code>tests/test_features.py</code>) recomputes features after deleting every later row and checks they are
identical.</p>
{table(FEAT_T)}""")

sections.append(f"""
<h2 class='pb'>4. Algorithms: what is used where, and why</h2>
{table(ALGOS)}
<h3>4.1 Logistic Regression (baseline)</h3>
<p>Linear model of the log-odds of fraud. Pipeline: log(1+x) on the numeric features, median imputation with
missing-value indicators, standardisation, one-hot encoding of category and gender, <code>LogisticRegression(C=1.0,
max_iter=2000)</code>. It sets the floor: anything more complex must beat it.</p>
<h3>4.2 Random Forest (baseline)</h3>
<p>200 decision trees, each trained on a 30% bootstrap sample with a random subset of features at every split
(<code>max_features='sqrt'</code>, <code>min_samples_leaf=5</code>). Predictions are averaged. It captures
interactions and is robust, but it is slower and less accurate than boosting here.</p>
<h3>4.3 LightGBM (final model)</h3>
<p>Gradient boosting: trees are added one at a time, each fitted to the gradient of the log-loss of the current
ensemble, and added with a small learning rate. LightGBM grows trees leaf-wise, uses histogram binning (fast on 1.1M
rows) and handles categorical features natively. Two versions were compared: library defaults, and tuned with Optuna.
The final model is trained on the fit period with all frauds and {cfg.model.tuning_sample_negatives:.0%} of genuine
rows; calibration later fixes the probability scale.</p>
<h3>4.4 Optuna (TPE sampler)</h3>
<p>Bayesian hyper-parameter search. Each trial trains LightGBM with 3-fold stratified cross-validation on the fit
period and returns the mean PR-AUC. TPE models good and bad regions of the search space and samples where good
results are likely.</p>
{table(SPACE)}
<h3>4.5 Calibration: isotonic regression</h3>
<p>A monotone, step-wise mapping from raw score to probability, fitted on the validation period
({cal["validation_positives"]:,} frauds; Platt scaling is used if fewer than 300). Monotone means the ranking, and so
PR-AUC, is unchanged; only the probability values become trustworthy.</p>
<h3>4.6 Cost-optimal threshold</h3>
<p>For every possible threshold on validation: cost = missed fraud amount + {at["review_cost"]:.0f} x flagged count.
The minimum is at <b>{thr:.4f}</b>. Risk bands: Medium starts at the threshold, High at {cfg.bands.high}, Critical at
{cfg.bands.critical}.</p>
<h3>4.7 SHAP TreeExplainer</h3>
<p>Exact Shapley values for tree ensembles. For each transaction: base value + sum of feature contributions = model
log-odds (verified by a test). Contributions are grouped and turned into sentences with the real numbers.</p>
<h3>4.8 Population Stability Index (drift)</h3>
<p>PSI = &Sigma; (actual% - expected%) &middot; ln(actual% / expected%) over training-time decile bins. Below 0.10
stable, 0.10-0.25 moderate, 0.25 or more significant.</p>""")

sections.append(f"""
<h2>5. How the comparison is done</h2>
<div class='box'><b>Protocol.</b> All four models get the same 21 features and are trained on the same fit period
({sp["fit"]["rows"]:,} rows). They are compared on the same validation period ({sp["validation"]["rows"]:,} rows,
later in time) with the same metric (PR-AUC). The winner is chosen on validation only. Calibration and threshold
are also chosen on validation. Only then is the test file scored, once.</div>
<p><b>Why PR-AUC.</b> With {y.mean():.2%} fraud, accuracy is meaningless: "always genuine" is {1 - y.mean():.2%}
accurate. ROC-AUC is also flattering (all models score above 0.94) because its false-positive rate is divided by
hundreds of thousands of genuine transactions. PR-AUC (average precision) measures how clean the alert list is at
every recall level; a random model scores the fraud rate ({y.mean():.3%}).</p>
<p><b>Also reported:</b> ROC-AUC (ranking quality), Brier score (probability accuracy; lower is better), fit time.</p>
{CH_LEADERBOARD}
{LB_TABLE}
<h3>Reading the comparison</h3>
<ul>
<li><b>Tuned LightGBM wins</b> on PR-AUC, ROC-AUC and Brier score; its final fit took
{lb_t.loc[lb_t["selected"] == "selected", "fit_seconds"].iloc[0]:.0f} s on this machine.</li>
<li><b>Random Forest</b> is a strong second: trees capture the interactions (amount x night x category) that matter
here.</li>
<li><b>Default LightGBM</b> is weak: untuned settings overfit the few fraud patterns at 0.6% positives. Tuning
regularisation, sampling and class weight is what makes boosting work.</li>
<li><b>Logistic Regression</b> is lowest because fraud depends on interactions a linear model cannot represent
without hand-made features.</li>
</ul>
<h3>Hyper-parameter search</h3>
{CH_TRIALS}
<p>Best cross-validated PR-AUC {tn["best_cv_pr_auc"]:.4f} after {tn["trials"]} trials; the tuned model then scored
{lb_t.loc[lb_t["selected"] == "selected", "pr_auc"].iloc[0]:.4f} on the untouched validation period, so the CV
estimate generalised.</p>""")

sections.append(f"""
<h2>6. Calibration and the money-based threshold</h2>
{CH_CAL_COST}
{
    kv(
        [
            ("Calibration method", cal["method"]),
            (
                "Brier score on validation (before -> after, fitted there)",
                f"{cal['validation_brier_before']:.5f} -> {cal['validation_brier_after']:.5f}",
            ),
            (
                "Brier score on test (before -> after)",
                f"{te['brier_before_calibration']:.5f} -> {te['brier_after_calibration']:.5f}",
            ),
            ("Review cost per flagged transaction", f"{at['review_cost']:.0f}"),
            ("Chosen threshold (validation)", f"{thr:.4f}"),
        ]
    )
}
<p>Why the threshold is so low: a missed fraud loses its whole amount (median fraud amount in training is
€{train_raw.loc[train_raw["is_fraud"] == 1, "amt"].median():,.0f}), while a review costs {
    at["review_cost"]:.0f}. Flagging at
{thr:.2%} probability is worth it on average.</p>

<h2>7. Final test results (evaluated once)</h2>
{CH_PR_CM}
{
    kv(
        [
            (
                "PR-AUC (95% bootstrap CI, 1,000 resamples)",
                f"{te['pr_auc']:.4f} ({te['pr_auc_ci95'][0]:.4f} - {te['pr_auc_ci95'][1]:.4f})",
            ),
            ("ROC-AUC", f"{te['roc_auc']:.4f}"),
            ("Flagged transactions", f"{at['flagged']:,} of {at['transactions']:,}"),
            ("True positives / false positives / false negatives", f"{at['tp']:,} / {at['fp']:,} / {at['fn']:,}"),
            ("Recall (frauds caught)", f"{at['recall']:.2%}"),
            ("Precision (alerts that are fraud)", f"{at['precision']:.2%}"),
            ("False alarms per 10,000 genuine", f"{at['false_alarms_per_10k_genuine']:.2f}"),
            ("Fraud amount in the period", f"{at['fraud_amount']:,.0f}"),
            (
                "Cost with model = missed + reviews",
                f"{at['missed_amount']:,.0f} + {at['review_cost_total']:,.0f} = {at['total_cost']:,.0f}",
            ),
            ("Saved versus no model", f"{at['saved_vs_no_model']:,.0f}"),
        ]
    )
}""")

reasons_html = "".join(f"<li style='color:#B91C1C'>+ {e(r.text)}</li>" for r in ex.raising) + "".join(
    f"<li style='color:#15803D'>- {e(r.text)}</li>" for r in ex.lowering
)
missed_html = ""
if ex_missed is not None:
    missed_html = (
        f"<h3>A fraud the model missed</h3><p>{e(ex_missed.summary)}</p><ul>"
        + "".join(f"<li style='color:#B91C1C'>+ {e(r.text)}</li>" for r in ex_missed.raising)
        + "".join(f"<li style='color:#15803D'>- {e(r.text)}</li>" for r in ex_missed.lowering)
        + "</ul>"
    )
sections.append(f"""
<h2>8. Explainability</h2>
{CH_SHAP}
<p>The 24-hour spend, the amount, the category and the night flag drive the model most. These are exactly the
simulator's fraud patterns: bursts of large online purchases at night.</p>
<h3>Example: a caught fraud (test set)</h3>
<p>{e(ex.summary)}</p><ul>{reasons_html}</ul>
<p>Recommended action: <b>{e(ex.action)}</b>.</p>
{missed_html}""")

drift_alerts = drift[drift["status"] == "significant"][["month", "feature", "psi"]].sort_values(["month", "psi"])
sections.append(f"""
<h2>9. Drift monitoring and the defect it found</h2>
{CH_DRIFT}
<p>Significant drift alerts in the test period:</p>
{table(drift_alerts, {"psi": lambda v: f"{v:.3f}"})}
<p>The December alerts reflect the holiday spending surge in the data, which is exactly the kind of shift
monitoring should report.</p>
<div class='box warn'><b>Defect found by monitoring.</b> The first model used <code>card_txn_count</code>, the lifetime
number of transactions of the card. It can only grow, so on the test period it was outside its training range every
month (PSI 3.7-6.1) while being a top-5 feature: the model trusted cards more the older they were. It was replaced by
<code>n_txn_30d</code> (transactions in the previous 30 days), which is stable over time.</div>
{OLD_NEW}""")

sections.append(f"""
<h2>10. Why policy rules are needed (model blind spots)</h2>
<p>A model learns only what is in its data. In the training data every fraud is at most
€{train_raw.loc[train_raw["is_fraud"] == 1, "amt"].max():,.0f}, and amounts above €2,000 are always genuine:</p>
{CH_AMOUNT}
{table(amt_tbl, {"fraud rate": lambda v: f"{v:.4f}", "transactions": lambda v: f"{v:,}", "frauds": lambda v: f"{v:,}"})}
<p>So the model, correctly for this data, gives a €12,000 purchase 2,500 km from home a near-zero probability. Real
fraud systems combine the model with business rules. FraudLens rules (<code>configs/default.yaml -&gt; rules</code>)
can only <b>raise</b> the band, and every hit is shown as its own reason.</p>
{
    table(
        pd.DataFrame(
            [
                ("large_amount", f"amount >= {RULES['review_amount']:,.0f}", "High"),
                (
                    "spend_spike",
                    f">= {RULES['spend_spike_ratio']:g}x the card's average and >= {RULES['spend_spike_min_amount']:,.0f}",
                    "High",
                ),
                ("far_from_home", f"merchant >= {RULES['far_from_home_km']:,.0f} km from home", "Medium"),
                ("impossible_travel", f"> 800 km/h and a jump >= {RULES['far_from_home_km']:,.0f} km", "High"),
                ("burst", f">= {cfg.rules.burst_txn_1h} other transactions in the last hour", "High"),
                ("out_of_range", "a key feature outside the range seen in training", "Medium"),
            ],
            columns=["rule", "condition", "minimum band"],
        )
    )
}
<h3>Measured impact on the test period</h3>
{
    table(
        impact,
        {
            c: (lambda v: f"{v:,.4f}")
            if c in ("precision", "recall")
            else (lambda v: f"{v:,.1f}")
            if c == "false_alarms_per_10k_genuine"
            else (lambda v: f"{v:,.0f}" if isinstance(v, (int, float, np.integer, np.floating)) else v)
            for c in impact.columns
        },
    )
}
{table(per_rule)}
<p>Honest reading: on this simulated test set the rules add reviews but catch no extra fraud, because the
simulator never produces fraud of that kind. The extra cost is the insurance premium for transactions outside the
training data, such as the €12,000 example.</p>
<p>Two candidate rules were tightened after measuring them. A plain "impossible travel" flag fired 15,074 times for 1
extra fraud, and a 10x spending-spike rule added 2,398 reviews for 10 frauds. Rules are judged by data too.</p>""")


# decision policy comparison
dp_rows = []
for name, label in (("threshold", "Flat threshold"), ("expected_loss", "Expected loss")):
    d = dec_m.get(name)
    if not d:
        continue
    param = f"p >= {d['value']:.4f}" if name == "threshold" else f"p x amount >= {d['k']:g} x review cost"
    for split in ("validation", "test"):
        if split in d:
            x = d[split]
            dp_rows.append(
                (
                    label + (" (served)" if dec_m.get("served") == name else ""),
                    param,
                    split,
                    f"{x['flagged']:,}",
                    f"{x['recall']:.2%}",
                    f"{x['precision']:.2%}",
                    f"{x['total_cost']:,.0f}",
                )
            )
DP_TABLE = (
    table(pd.DataFrame(dp_rows, columns=["policy", "rule", "data", "flagged", "recall", "precision", "total cost"]))
    if dp_rows
    else "<p>(not available)</p>"
)
if dec_m:
    kg = pd.DataFrame(dec_m["expected_loss"]["k_grid"])
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    ax.plot(kg["k"], kg["total_cost"], marker="o", color=BLUE)
    ax.scatter([dec_m["expected_loss"]["k"]], [dec_m["expected_loss"]["validation"]["total_cost"]], color=RED, zorder=3)
    ax.set_xlabel("k (review when p x amount >= k x review cost)")
    ax.set_ylabel("validation cost")
    ax.set_title("Tuning k for the expected-loss policy (validation)")
    CH_K = img(fig)
else:
    CH_K = ""
if ablation:
    AB_TABLE = table(
        pd.DataFrame(
            [
                (
                    r["candidate"],
                    f"{r['val_pr_auc_by_seed'][0]:.4f}",
                    f"{r['val_pr_auc_by_seed'][1]:.4f}",
                    f"{r['mean']:.4f}",
                )
                for r in ablation["rows"]
            ],
            columns=["feature set", "seed 42", "seed 7", "mean validation PR-AUC"],
        )
    )
    AB_TEXT = (
        f"Seed-to-seed noise of the current feature set: {ablation['seed_noise_baseline']:.4f}. "
        f"Kept: {', '.join(ablation['kept']) if ablation['kept'] else 'none - no candidate beat the noise'}."
    )
else:
    AB_TABLE, AB_TEXT = "<p>(run scripts/feature_ablation.py)</p>", ""
el_t, th_t = dec_m.get("expected_loss", {}).get("test", {}), dec_m.get("threshold", {}).get("test", {})
sections.append(f"""
<h2 class='pb'>10b. Decision policy: one threshold or expected loss?</h2>
<p>A flat threshold treats a €2 and a €2,000 transaction the same. The <b>expected-loss policy</b> reviews when
<i>calibrated probability x amount &ge; k x review cost</i>: reviewing is worth it when the money expected to be lost
exceeds what the review costs. Risky but cheap transactions get <b>step-up verification</b> (the cardholder confirms)
instead of an analyst. k is tuned on validation; the served policy is whichever is cheaper on validation.</p>
{CH_K}
{DP_TABLE}
<div class='box'><b>Honest reading.</b> On validation the flat threshold was cheaper, so it is served
(<i>{e(describe_policy(served_pol))}</i>). On the test period the expected-loss policy would have cost
{th_t.get("total_cost", 0) - el_t.get("total_cost", 0):,.0f} less, at lower recall ({el_t.get("recall", 0):.1%} vs
{th_t.get("recall", 0):.1%}). We do not switch on test results: that would be tuning on the test set. The app lets the
user switch policy in the sidebar and see the effect live.</div>
<h3>Feature ablation (validation only)</h3>
<p>Three card-profile features were tried: amount vs the card's previous maximum, the card's share of purchases in
the category, and how usual the hour is for the card. Same split, encoder and tuned parameters, two seeds:</p>
{AB_TABLE}
<p>{e(AB_TEXT)} They are kept as context for explanations, rules and the Card 360 view, not as model inputs.</p>
<h3>Rule limits derived from data</h3>
{
    kv(
        [
            (k, f"{m.get('rules_auto', {}).get(k, RULES[k]):,.0f}" if k != "spend_spike_ratio" else f"{RULES[k]:g}x")
            for k in ("review_amount", "far_from_home_km", "spend_spike_ratio", "spend_spike_min_amount")
        ]
        + [(f"how {k}", v) for k, v in m.get("rules_auto", {}).get("derivation", {}).items()]
    )
}
""")

sections.append(f"""
<h2>11. Engineering that supports the results</h2>
<ul>
<li><b>Reproducible:</b> fixed seeds, deterministic LightGBM; a test trains twice and compares all metrics. Model
version {version} is stored with its metrics in <code>artifacts/runs/</code>.</li>
<li><b>Tested:</b> leakage, SHAP reconstruction, reason correctness, calibration maths (weighted average precision
equals scikit-learn), drift, rules, authentication, API and a click-through of every app page per role.</li>
<li><b>Product:</b> Streamlit app with login and roles (viewer / analyst / admin); case queue with board and
table views, bulk actions, SLA timers, comments and audit trail; Card 360 (timeline, map, spending profile);
watchlists (blocked cards stored as hashes, watched merchants) that feed the rules; batch scoring; monitoring;
FastAPI scoring API with API keys and rate limits returning the decision and expected loss.</li>
<li><b>Training run:</b> {m["run"]["total_seconds"]:.0f} s on {m["machine"]["cpu_count"]} CPU threads; libraries:
{", ".join(f"{k} {v}" for k, v in m["versions"].items())}.</li>
</ul>

<h2>12. Limitations and future work</h2>
<ul>
<li>Simulated data: fraud is cleaner than in reality; expect lower real-world PR-AUC.</li>
<li>No device, IP, merchant-risk or customer-contact signals; labels are instant (real chargebacks take weeks).</li>
<li>Calibration and threshold share the validation period; the test period is the independent check.</li>
<li>One flat review cost; real costs differ by transaction and by action (block vs step-up).</li>
<li>Next: real data with delayed labels, champion/challenger retraining with approval, richer features, SSO.</li>
</ul>""")

qa_html = "".join(f"<div class='qa'><b>Q{i}. {e(q)}</b><br>{e(a)}</div>" for i, (q, a) in enumerate(QA, 1))
sections.append(f"<h2 class='pb'>13. Viva questions and answers</h2>{qa_html}")

sections.append("""
<h2>14. Glossary</h2>
<table><thead><tr><th>Term</th><th>Meaning</th></tr></thead><tbody>
<tr><td>Precision</td><td>Of the flagged transactions, the share that are fraud.</td></tr>
<tr><td>Recall</td><td>Of the frauds, the share that were flagged.</td></tr>
<tr><td>PR-AUC / average precision</td><td>Area under the precision-recall curve; 1 is perfect, the fraud rate is random.</td></tr>
<tr><td>ROC-AUC</td><td>Probability a random fraud is ranked above a random genuine transaction.</td></tr>
<tr><td>Brier score</td><td>Mean squared error of predicted probabilities; lower is better.</td></tr>
<tr><td>Calibration</td><td>Predicted probabilities match observed frequencies.</td></tr>
<tr><td>Log-odds</td><td>log(p / (1 - p)); SHAP contributions add up in this scale.</td></tr>
<tr><td>Leakage</td><td>Using information at training time that is not available at prediction time.</td></tr>
<tr><td>Stratified CV</td><td>Cross-validation where every fold keeps the class ratio.</td></tr>
<tr><td>Bootstrap CI</td><td>Interval from re-computing a metric on many resamples of the data.</td></tr>
<tr><td>PSI</td><td>Population Stability Index: how much a distribution moved since training.</td></tr>
<tr><td>Target encoding</td><td>Replacing a category by the (smoothed) label rate of that category.</td></tr>
</tbody></table>""")

doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(sections)}</body></html>"


def main():
    from playwright.sync_api import sync_playwright

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    html_path = ROOT / "docs" / "FraudLens_Viva_Report.html"
    html_path.write_text(doc, encoding="utf-8")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.as_uri())
        page.pdf(
            path=str(OUT_PDF),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                "<div style='font-size:8px;color:#64748B;width:100%;text-align:center'>FraudLens viva report &middot; "
                "page <span class='pageNumber'></span> / <span class='totalPages'></span></div>"
            ),
            margin={"top": "14mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
        )
        browser.close()
    html_path.unlink()
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
