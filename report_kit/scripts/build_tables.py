"""Hand-curated tables (literature, base paper, tech stack, features, pages, API, screenshots, inconsistencies,
open questions, viva Q&A). Numbers are read from fraudlens/artifacts and report_kit/_data at run time.

Run after extract_results.py and extra_facts.py:  fraudlens/.venv/Scripts/python.exe report_kit/scripts/build_tables.py
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pandas as pd

KIT = Path(__file__).resolve().parents[1]
FL = KIT.parent / "fraudlens"
T = KIT / "tables"
D = KIT / "_data"
M = json.loads((FL / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
X = json.loads((D / "extra_facts.json").read_text(encoding="utf-8"))
S = json.loads((D / "summary.json").read_text(encoding="utf-8"))
RUNS = pd.read_csv(T / "runs_comparison.csv")
import html

REFV = pd.read_csv(T / "references_verified_final.csv", keep_default_na=False).set_index("key")
REFV = REFV.apply(lambda col: col.map(lambda v: html.unescape(v) if isinstance(v, str) else v))
test, at = M["test"], M["test"]["at_threshold"]


def write(name: str, rows: list[dict]):
    with (T / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"tables/{name}: {len(rows)} rows")


def ref(key: str) -> dict:
    r = REFV.loc[key]
    return {"venue": r["container"], "volume": r["volume"], "issue": r["issue"], "pages": r["pages"], "doi": r["doi"], "year": r["year"]}


# ------------------------------------------------------------------ literature survey
TEAM = "team's own review: docs/Literature_Review_Fraud_Detection.docx"
NS = "NOT STATED in the abstract; full text not read"
lit = [
    ("Tanouz et al.", "tanouz2021", "Classifiers named in the abstract: logistic regression, random forest, Naive Bayes (the team's review also lists a decision tree: see INCONSISTENCIES)",
     "ULB card dataset (per " + TEAM + ")", "Random Forest best with 96.77% accuracy, 100% precision, 91.11% recall (as quoted in the team's review; UNVERIFIED against the paper, which is paywalled)",
     "Per the team's review: tiny balanced test set (~124 rows), preprocessing before the split, 'ROC-AUC' looks like balanced accuracy", "abstract (Semantic Scholar) + " + TEAM, "metadata VERIFIED (Crossref); numbers UNVERIFIED"),
    ("Bhattacharyya et al.", "bhattacharyya2011", "SVM, random forest and logistic regression compared (per " + TEAM + ")", "Real credit-card transactions (per team's review)",
     "Random forest strongest overall, with a precision-recall trade-off (per team's review)", "NOT STATED (abstract not retrievable; full text not read)", TEAM, "metadata VERIFIED (Crossref); content from team's review"),
    ("Randhawa et al.", "randhawa2018", "Standard classifiers, then hybrid AdaBoost and majority voting", "A public credit-card dataset, then a real-world dataset from a financial institution, with noise added",
     "Majority voting gives good accuracy in detecting fraud (no numbers in the abstract)", NS, "abstract (Semantic Scholar)", "VERIFIED (Crossref)"),
    ("Dal Pozzolo et al.", "dalpozzolo2014", "Practitioner view: imbalance, concept drift and limited investigator capacity; recommends ranking-based metrics such as precision among top alerts (per team's review)", "NOT STATED",
     "NOT STATED", "NOT STATED", TEAM, "metadata VERIFIED (Crossref); content from team's review"),
    ("Dal Pozzolo et al.", "dalpozzolo2015cal", "Under-sampling shifts predicted probabilities, so they must be re-calibrated (per team's review)", "NOT STATED", "NOT STATED", "NOT STATED", TEAM, "metadata VERIFIED (Crossref); content from team's review"),
    ("Dal Pozzolo et al.", "dalpozzolo2018", "Realistic modelling of a fraud-detection system (delayed supervised information, alert-based performance measures) and a new learning strategy", "Data from an industrial partner (abstract)",
     "NOT STATED in the part of the abstract read", NS, "abstract (Semantic Scholar)", "VERIFIED (Crossref); not cited in the repo's docs"),
    ("He and Garcia", "hegarcia2009", "Survey of learning from imbalanced data: sampling, cost-sensitive and algorithm-level remedies (per team's review)", "N/A (survey)", "N/A (survey)", "NOT STATED", TEAM, "metadata VERIFIED (Crossref); content from team's review"),
    ("Chawla et al.", "chawla2002", "SMOTE: synthetic minority over-sampling combined with under-sampling the majority class", "NOT STATED in the abstract",
     "Better classifier performance in ROC space than under-sampling alone (abstract)", "NOT STATED in the abstract", "abstract (Semantic Scholar)", "VERIFIED (Crossref)"),
    ("Fernandez et al.", "fernandez2018", "15-year review of SMOTE and its extensions", "N/A (review)", "SMOTE is a de facto standard for imbalanced learning (abstract)",
     "Can add noise and gives little gain for strong tree ensembles on some datasets (per team's review)", "abstract + " + TEAM, "VERIFIED (Crossref)"),
    ("Saito and Rehmsmeier", "saito2015", "Compares ROC and precision-recall plots on imbalanced data", "Imbalanced datasets (bioinformatics context in the abstract)", "ROC plots can be deceptive on imbalanced data; PR plots are more informative (abstract)", NS, "abstract (Semantic Scholar)", "VERIFIED (Crossref)"),
    ("Davis and Goadrich", "davis2006", "Relationship between precision-recall and ROC curves: a curve dominates in ROC space iff it dominates in PR space (per team's review)", "N/A (theory)", "NOT STATED", "NOT STATED", TEAM, "metadata VERIFIED (Crossref); content from team's review"),
    ("Kaufman et al.", "kaufman2012", "Formal definition of data leakage and how to detect and avoid it (per team's review)", "N/A", "N/A", "NOT STATED", TEAM, "metadata VERIFIED (Crossref); content from team's review"),
    ("Ribeiro et al.", "ribeiro2016", "LIME: explains one prediction of any classifier with a local interpretable model", "Text and image classifiers (not card fraud)", "NOT STATED in the part of the abstract read", "Explanations of PCA-anonymised features are not business-interpretable (team's note)", "abstract + " + TEAM, "VERIFIED (Crossref)"),
    ("Breiman", "breiman2001", "Random forests: combination of tree predictors; generalisation error converges as trees grow", "N/A", "N/A", NS, "abstract (Semantic Scholar)", "VERIFIED (Crossref)"),
    ("Chen and Guestrin", "chen2016", "XGBoost: scalable tree boosting with a sparsity-aware algorithm", "NOT STATED in the part of the abstract read", "NOT STATED in the part of the abstract read", NS, "abstract (Semantic Scholar)", "VERIFIED (Crossref); used in the original notebook, not in FraudLens"),
    ("Akiba et al.", "akiba2019", "Optuna: define-by-run hyper-parameter optimisation framework with pruning", "N/A", "N/A", NS, "abstract (Semantic Scholar)", "VERIFIED (Crossref); the tool used for tuning in fraudkit/train.py"),
    ("Niculescu-Mizil and Caruana", "niculescu2005", "Probability calibration of supervised learners (title; abstract not retrieved)", "NOT STATED", "NOT STATED", "NOT STATED", "title only", "VERIFIED (Crossref); content NOT read"),
    ("Lucas et al.", "arxiv1909", "Hidden Markov models over sequences of amounts and elapsed time give extra features for card-fraud detection", "NOT STATED in the part of the abstract read", "NOT STATED", "NOT STATED", "abstract (arXiv API)", "VERIFIED (arXiv record); preprint"),
    ("Stotsky", "arxiv2503", "Moving-window confidence-interval method and a fraud-risk scoring method for detecting fraud attacks", "A card-transaction dataset (not named in the part read)", "NOT STATED", "NOT STATED", "abstract (arXiv API)", "VERIFIED (arXiv record); preprint, not peer reviewed"),
]
rows = []
for i, (auth, key, method, data, result, limit, src, ver) in enumerate(lit, 1):
    r = ref(key) if key in REFV.index else {"venue": "arXiv", "volume": "", "issue": "", "pages": "", "doi": "", "year": 2019 if key == "arxiv1909" else 2025}
    rows.append({"no": i, "citation_key": key, "authors": auth, "year": r["year"], "venue": r["venue"], "method": method, "dataset": data, "result": result,
                 "limitation": limit, "summary_source": src, "verification": ver, "doi": r["doi"]})
write("literature_survey.csv", rows)

# ------------------------------------------------------------------ base paper vs ours
write("base_paper_vs_ours.csv", [
    {"setting": "Paper as reported [Tanouz 2021]", "dataset": "ULB (Kaggle creditcard)", "test_rows": "~124, balanced", "precision_pct": 100.00, "recall_pct": 91.11, "false_alarms": 0, "source": "docs/Literature_Review_Fraud_Detection.docx Table 1; UNVERIFIED against the paper (paywalled)"},
    {"setting": "Team's re-implementation, paper-style (Random Forest)", "dataset": "ULB", "test_rows": "247, balanced", "precision_pct": 94.68, "recall_pct": 86.41, "false_alarms": 5, "source": "same table; notebook has no saved outputs so NOT re-run in this kit"},
    {"setting": "Random Forest, under-sampled training, real test", "dataset": "ULB", "test_rows": "56,962 (98 fraud)", "precision_pct": 4.30, "recall_pct": 91.84, "false_alarms": 2004, "source": "same table; NOT re-run in this kit"},
    {"setting": "Random Forest + SMOTE, real test", "dataset": "ULB", "test_rows": "56,962 (98 fraud)", "precision_pct": 88.30, "recall_pct": 84.69, "false_alarms": 11, "source": "same table; NOT re-run in this kit"},
    {"setting": "FraudLens tuned LightGBM at the validated threshold", "dataset": "Sparkov simulated (Kaggle kartik2112)", "test_rows": f"{M['split']['test']['rows']:,} ({M['split']['test']['frauds']:,} fraud)",
     "precision_pct": round(100 * at["precision"], 2), "recall_pct": round(100 * at["recall"], 2), "false_alarms": at["fp"], "source": "fraudlens/artifacts/metrics.json -> test.at_threshold (different dataset: NOT directly comparable to the rows above)"},
])

# ------------------------------------------------------------------ tech stack
v = M["versions"]
mach = M["machine"]
stack = [
    ("Language", "Python", v["python"], "metrics.json versions"), ("Data", "pandas", v["pandas"], "metrics.json versions"), ("Data", "numpy", v["numpy"], "metrics.json versions"),
    ("ML", "scikit-learn", v["scikit-learn"], "metrics.json versions"), ("ML", "lightgbm", v["lightgbm"], "metrics.json versions"), ("ML", "shap", v["shap"], "metrics.json versions"),
    ("ML", "optuna", v["optuna"], "metrics.json versions"), ("UI", "streamlit", v["streamlit"], "metrics.json versions"), ("UI", "plotly", v["plotly"], "metrics.json versions"),
    ("Core", "joblib", "1.6.0", "fraudlens/requirements.txt"), ("Core", "kagglehub", "1.0.2", "fraudlens/requirements.txt"), ("Core", "pydantic", "2.14.1", "fraudlens/requirements.txt"),
    ("Core", "pyyaml", "6.0.3", "fraudlens/requirements.txt"), ("Core", "rich", "15.0.0", "fraudlens/requirements.txt"), ("Test", "pytest", "9.1.1", "fraudlens/requirements.txt"),
    ("API", "fastapi", "0.153.0", "fraudlens/requirements-api.txt"), ("API", "uvicorn", "0.54.0", "fraudlens/requirements-api.txt"), ("API", "httpx", "0.28.1", "fraudlens/requirements-api.txt"),
    ("Quality", "ruff", "0.17.1", "fraudlens/requirements-dev.txt and .github/workflows/fraudlens-ci.yml"), ("Tooling", "playwright (screenshots)", "1.64.0", "fraudlens/requirements-dev.txt"),
    ("Tooling", "matplotlib", "3.11.2", "fraudlens/requirements-dev.txt"), ("Tooling", "pypdfium2", "5.14.0", "fraudlens/requirements-dev.txt"),
    ("Deploy", "huggingface_hub (deploy script)", "2.4.0", "fraudlens/requirements-deploy.txt"), ("Deploy", "Docker base image", "python:3.11-slim", "fraudlens/Dockerfile"),
    ("Deploy", "Docker engine", "29.6.1", "command: docker info --format '{{.ServerVersion}}' (run in this session)"),
    ("Storage", "SQLite (runtime library)", "3.50.4", "command: python -c 'import sqlite3; print(sqlite3.sqlite_version)'"), ("Storage", "pyarrow (parquet)", "26.0.0", "command: importlib.metadata.version('pyarrow') in fraudlens/.venv"),
    ("Diagrams", "Graphviz dot", "15.0.0", "command: dot -V"), ("Hardware", "CPU threads used for training", str(mach["cpu_count"]), "metrics.json machine"),
    ("Hardware", "Processor", mach["processor"], "metrics.json machine"), ("OS", "Training machine platform", mach["platform"], "metrics.json machine"),
]
write("tech_stack.csv", [{"category": a, "component": b, "version": c, "source": d} for a, b, c, d in stack])

# ------------------------------------------------------------------ features
F = "fraudlens/fraudkit/features.py"
feat = [
    ("amt", "raw", "Transaction amount (currency units of the dataset, shown as EUR)", "column amt", True),
    ("log_amt", "amount", "log(1 + amt)", f"{F}:159", True), ("hour", "time", "Hour of day of the transaction (0-23)", f"{F}:160", True), ("day_of_week", "time", "Day of week (Monday = 0)", f"{F}:161", True),
    ("is_night", "time", "1 if hour is in {22, 23, 0, 1, 2, 3}", f"{F}:14,162", True), ("dist_home_km", "geo", "Haversine distance (km) between the cardholder's home (lat, long) and the merchant (merch_lat, merch_long)", f"{F}:81-84,163", True),
    ("age", "demographic", "(transaction time - date of birth) in days / 365.25", f"{F}:164", True), ("log_city_pop", "demographic", "log(1 + city_pop)", f"{F}:165", True),
    ("merchant_te", "merchant", "Smoothed target encoding of merchant: (sum_fraud + m * prior) / (count + m), m = 50; out-of-fold (5 folds) for fit rows, fitted on the fit split only", f"{F}:284-320", True),
    ("n_txn_1h", "velocity", "Number of EARLIER transactions of the card in the previous 1 hour", f"{F}:92-99,209-212,237", True), ("n_txn_24h", "velocity", "Earlier transactions of the card in the previous 24 hours", f"{F}:211,238", True),
    ("amt_sum_24h", "velocity", "Sum of amounts of the card's earlier transactions in the previous 24 hours", f"{F}:211,240", True), ("n_txn_7d", "velocity", "Earlier transactions of the card in the previous 7 days", f"{F}:250,255", True),
    ("n_txn_30d", "velocity", "Earlier transactions of the card in the previous 30 days (replaced the lifetime counter card_txn_count, which drifted)", f"{F}:212,239", True),
    ("avg_amt_1d", "spending window", "Mean amount of the card's earlier transactions in the last 1 day (NaN if none)", f"{F}:256", True), ("avg_amt_7d", "spending window", "Same, 7 days", f"{F}:257", True),
    ("avg_amt_30d", "spending window", "Same, 30 days", f"{F}:258", True), ("amt_to_avg_30d", "spending window", "amt / avg_amt_30d", f"{F}:259", True),
    ("amt_to_card_mean", "card profile", "amt / mean of all the card's earlier amounts", f"{F}:169-174", True), ("amt_to_cat_mean", "card profile", "amt / mean of the card's earlier amounts in the same category", f"{F}:176-182", True),
    ("amt_log_z", "card profile", "(log(1+amt) - mean) / std of log(1+amount) over the card's earlier transactions (needs >= 2 earlier rows and std > 1e-6)", f"{F}:260-267", True),
    ("first_time_category", "card profile", "1 if this is the card's first purchase in this category", f"{F}:184", True), ("hours_since_prev", "sequence", "Hours since the card's previous transaction", f"{F}:187-189", True),
    ("geo_speed_kmh", "geo", "Distance between the previous and the current merchant location / max(hours since previous, 1/60 h)", f"{F}:191-196", True),
    ("impossible_travel", "geo", "1 if geo_speed_kmh > 800 and the jump > 50 km", f"{F}:15,198", True), ("small_txn_24h", "card testing", "Number of the card's earlier transactions below 5 in the last 24 hours", f"{F}:16,247,252,268", True),
    ("category", "categorical", "Merchant category (14 levels in the training file)", "column category", True), ("gender", "categorical", "Cardholder gender (see ethics note)", "column gender", True),
]
assert len(feat) == len(M["features"]) == 28 and {f[0] for f in feat} == set(M["features"]), "feature catalogue out of sync with metrics.json"
imp = M["global_importance"]["mean_abs_shap"]
rank = {k: i + 1 for i, k in enumerate(imp)}
write("features_catalog.csv", [{"feature": a, "group": b, "definition": c, "source": d, "mean_abs_shap": round(imp[a], 4), "importance_rank": rank[a]} for a, b, c, d, _ in feat])

# ------------------------------------------------------------------ pages and roles (parsed from app/common.py PAGES), API endpoints
src = (FL / "app" / "common.py").read_text(encoding="utf-8")
tree = ast.parse(src)
pages = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PAGES")
import sys  # noqa: E402

sys.path.insert(0, str(FL))
from fraudkit.auth import PERMISSIONS  # noqa: E402

write("pages_roles.csv", [{"section": s, "page": t, "file": f"fraudlens/app/{p}", "permission_to_open": perm, "roles_that_can_open": ", ".join(r for r in ("viewer", "analyst", "admin") if r in PERMISSIONS[perm])} for s, p, t, _i, perm in pages])
ep = [
    ("GET", "/v1/health", "none", "liveness probe", "api/main.py:235"), ("GET", "/v1/ready", "none", "model loaded + model version", "api/main.py:240"),
    ("GET", "/v1/model", "X-API-Key", "model version, threshold, metrics summary", "api/main.py:250"), ("POST", "/v1/score", "X-API-Key", "score up to 1,000 transactions; returns probability, band, decision, risk score, reasons", "api/main.py:270"),
    ("GET", "/health", "none", "legacy route kept for old clients", "api/main.py:329"), ("POST", "/score", "X-API-Key", "legacy route (returns a bare list)", "api/main.py:334"),
]
write("api_endpoints.csv", [{"method": a, "path": b, "auth": c, "purpose": d, "source": e} for a, b, c, d, e in ep])

# ------------------------------------------------------------------ screenshots
open_to = {p.split("/")[-1].split("_", 1)[0] if False else t: ", ".join(r for r in ("viewer", "analyst", "admin") if r in PERMISSIONS[perm]) for _s, p, t, _i, perm in pages}
ALL, AA = open_to["Dashboard"], open_to["Batch scoring"]
shots = [
    ("0_login.png", "Sign in", "any visitor", "(no sign-in needed to see this page)", "Sign-in form shown before any page. On a fresh database it becomes a first-run setup that creates the first administrator.", "no: caption from fraudlens/app/common.py _login_screen"),
    ("1_dashboard.png", "Dashboard", "admin (demo.admin)", ALL, "KPI tiles for the held-out test period (555,719 scored, 2,463 flagged, recall 95.9%, precision 83.5%, 7.3 false alarms per 10k) with daily alerts, decision mix (approve 553,256, review 512, block 1,951), flagged-by-category, hour/weekday heat map and top merchants. Policy rules are not included in these tiles.", "yes"),
    ("2_check_transaction.png", "Check a transaction", "admin (demo.admin)", ALL + " (entering a transaction and what-if need analyst or admin)", "The highest-risk real test transaction (EUR 856.30, misc_net, 22:57): risk score 999, Critical, action Block, with red reasons (+9.35 large amount, +7.55 amount 5.4x the 30-day average, +4.42 burst of transactions, +4.27 night) and green reasons.", "yes"),
    ("2b_what_if.png", "Check a transaction: what-if", "admin (demo.admin)", "analyst, admin", "What-if panel: edit amount, hour of day or merchant distance and re-score the same transaction with the card's real history.", "no: caption from fraudlens/app/views/2_Check_Transaction.py"),
    ("2c_policy_rules.png", "Check a transaction: policy rules", "admin (demo.admin)", "analyst, admin", "A typed transaction of EUR 12,000 at 2,500 km from home: the model alone says Low (0.01%); five policy rules raise the band to High and the action to Review (risk score 231). This is the case that motivated the rule layer.", "yes"),
    ("3_batch_scoring.png", "Batch scoring", "admin (demo.admin)", AA, "Upload a CSV (or score the 500-row sample) to get a risk-ranked table with a reason per row, a CSV download and an option to send flagged rows to the case queue.", "no: caption from fraudlens/app/views/3_Batch_Scoring.py and a live run in this session"),
    ("4_case_queue.png", "Case queue", "admin (demo.admin)", ALL + " (acting on cases needs analyst or admin; clearing the queue needs admin)", "Kanban board of 24 new and 1 closed demo cases with SLA and assignee, a case detail with reasons, Confirm fraud / Mark genuine / Escalate, comments, audit trail and analyst-feedback charts (100% agreement is based on one decision).", "yes"),
    ("5_model_report.png", "Model report", "admin (demo.admin)", ALL, "Validation leaderboard, test PR-AUC 0.9765 with CI, reliability and cost curves, the calibration comparison, threshold vs expected-loss policy, measured effect of policy rules, global SHAP importance, data card and the limitations text.", "yes"),
    ("6_training_lab.png", "Training Lab", "admin (demo.admin)", ALL + " (the Retrain button needs admin)", "Latest training run read from artifacts: stage timeline, Optuna trial history, final leaderboard, chosen hyper-parameters, machine and library versions, stage durations.", "no: caption from fraudlens/app/views/6_Training_Lab.py"),
    ("7_monitoring.png", "Monitoring", "admin (demo.admin)", ALL, "Monthly PSI drift per feature against the training reference: 4 features significant and 5 moderate in 2020-12 (n_txn_7d PSI 1.030), score PSI 0.007, plus weekly precision/recall/flag rate.", "yes"),
    ("8_admin.png", "Admin", "admin (demo.admin)", "admin", "Admin tabs Users, API keys, Watchlists, Audit log and System; the Users tab shows add-user, role change, disable and temporary-password reset.", "yes"),
    ("9_card_360.png", "Card 360", "admin (demo.admin)", ALL + " (blocking a card needs analyst or admin)", "One card (****5105, Hartford AL): 736 transactions, EUR 53.2k spend, 9 frauds, 8 flagged, 1 case, with a time scatter, map of locations, spend by category, usual hours and recent transactions.", "yes"),
]
write("screenshots_captions.csv", [{"file": f"figures/screenshots/{a}", "page": b, "signed_in_as_in_the_image": c, "roles_that_can_open_this_page": d, "caption": e, "image_inspected_for_caption": f} for a, b, c, d, e, f in shots])

# ------------------------------------------------------------------ inconsistencies
r5 = RUNS.set_index("model_version")
inc = [
    ("PROGRESS.md M11", "Risk score: 'genuine median 25'", f"Recomputed from artifacts/test_scored.parquet with FraudModel.risk_score: genuine median {X['risk_score']['genuine_median']:.0f} (_data/extra_facts.json)", "minor"),
    ("fraudkit/explain.py:73", "Comment says the 0-999 risk score is a 'rank against the validation period'", "fraudkit/score.py:29-39: the score is a log-odds scale (p=1e-6 -> 0, p=0.999 -> 999), not a rank (the rank version was replaced, PROGRESS M11)", "minor (stale comment)"),
    ("app/views/5_Model_Report.py:309-312 (visible in screenshot 5)", "Limitations text says 'Drift is not monitored' and that 'Isotonic calibration and the threshold are both fitted on' validation",
     "app/views/7_Monitoring.py implements monthly PSI drift; metrics.json calibration.method = platt. Alerting/retraining automation really is absent, but the text is stale", "medium (the app contradicts itself)"),
    ("fraudkit/features.py:65", "Comment: n_txn_30d 'is stationary'", f"tables/drift_test_period_by_month.csv: n_txn_30d is the top-PSI feature in 2020-07/08/10/11 (0.11-0.16, moderate) and PSI 0.517 in 2020-12 (significant); whole-test-period PSI is 0.0878 (stable)", "minor"),
    ("PROGRESS.md M3/M8/M9/M10 and README history", "Numbers quoted for earlier models (e.g. M10: k = 3, validation cost 8,838 vs 9,338; M9: 2,548 -> 3,159 flags; M8: threshold 0.0238)",
     f"The final artifacts differ because the model changed: k = {M['decision']['expected_loss']['k']:g}, validation cost {M['decision']['threshold']['validation']['total_cost']:,.2f} vs {M['decision']['expected_loss']['validation']['total_cost']:,.2f}, policy rules 2,463 -> 2,871 flags (tables/policy_rules_impact_test.csv). Use only the final-artifact numbers in the report", "medium (superseded numbers)"),
    ("PROGRESS.md M3", "'Full run ... 612 s'", f"Archived first run 20261020-153708-3c73d465 has total_seconds {r5.loc['20261020-153708-3c73d465', 'wall_clock_seconds']}", "minor"),
    ("Final run 20261026-184402-d37e9778", f"total_seconds {r5.loc['20261026-184402-d37e9778', 'wall_clock_seconds']:,.1f}", f"Archived run 20261025-233115-3abe0ebd has the same test metrics (PR-AUC {r5.loc['20261025-233115-3abe0ebd', 'test_pr_auc']:.10f}) in {r5.loc['20261025-233115-3abe0ebd', 'wall_clock_seconds']:,.1f} s; the sum of Optuna trial seconds in the final run is {S['trials']['sum_trial_seconds']:,.0f} s, so the long time is real elapsed time (the README says the computer was asleep). The two model files have different hash suffixes (d37e9778 vs 3abe0ebd) although metrics agree to ~1e-15; the cause was NOT investigated", "minor (use 1,236 s for training time)"),
    ("artifacts/metrics.json test.at_policy.threshold and decision.*.test.threshold", "Value 0.5", "These are placeholders: fraudkit/decision.py:64 and train code call confusion_at(y, flags, amt, 0.5, review_cost) with 0/1 flags. The real threshold is 0.0837 (metrics.json threshold.value)", "minor (misleading field)"),
    ("README 'distinct probabilities'", "522,809 distinct probabilities on the test set", "fraudkit/train.py:710 counts distinct values AFTER rounding to 12 decimals; artifacts/test_scored.parquet has 555,719 distinct raw values (one per row)", "minor (definition)"),
    ("README / PROGRESS test count", "'79 tests passing' (PROGRESS M11) and 81 earlier today", "pytest today: 81 collected = 80 passed + 1 failed (tests/test_train_repro.py, SSLError reaching api.kaggle.com). The test needs network access; it is skipped in CI because the dataset is not cached there", "medium (report the failure honestly)"),
    ("docs/Literature_Review_Fraud_Detection.docx reference [4]", "Authors listed as '... G. V. P. Reddy, C. V. N. M. Praneeth, A. R. Kumar'; four classifiers incl. Decision Tree",
     "Crossref/Semantic Scholar list the order D. Tanouz, R. Raja Subramanian, D. Eswar, G. V. Parameswara Reddy, A. Ranjith Kumar, CH. V. N. M. Praneeth; the abstract names only logistic regression, random forest and Naive Bayes", "medium (fix the reference before submission)"),
    ("fraudlens/scripts/build_report.py section 15 (viva report) and PROGRESS M11", "arXiv 1909.01185 supports 'amount and elapsed time ... log z-score'; arXiv 2503.20477 supports 'circular time'",
     "The abstracts (arXiv API) say: 1909.01185 = HMM features from sequences of amounts / elapsed time; 2503.20477 = moving-window confidence intervals and fraud-risk scoring. Neither abstract mentions a log z-score or circular time encoding; full texts were not read -> UNVERIFIED", "medium"),
    ("Root README.md FraudLens paragraph", "'a six-page Streamlit app'", f"fraudlens/app/common.py PAGES lists {len(pages)} pages (tables/pages_roles.csv)", "minor (root README not edited, by instruction)"),
    ("docs/screenshots/*.png", "Signed-in user 'demo.admin'", "The screenshots were taken with an auto-created admin account (fraudlens/scripts/screenshots.py:66), so every page looks admin-only-capable; viewer/analyst views differ (tables/pages_roles.csv)", "info"),
]
write("inconsistencies.csv", [{"id": f"I{i}", "where": a, "claim_or_value": b, "evidence": c, "severity": d} for i, (a, b, c, d) in enumerate(inc, 1)])

# ------------------------------------------------------------------ viva Q&A
cmp_ = pd.read_csv(T / "decision_policy_comparison.csv")
cal = pd.DataFrame(M["calibration"]["comparison"]).set_index("method")
rows_v = pd.read_csv(T / "runs_comparison.csv")
lb = {r["model"]: r["pr_auc"] for r in M["leaderboard"]}
fr = M["split"]["test"]["fraud_rate"]
pim = pd.read_csv(T / "policy_rules_impact_test.csv").set_index("system")
extra_rev = int(pim.loc["Model policy + rules", "flagged"] - pim.loc["Model policy alone", "flagged"])
extra_cost = pim.loc["Model policy + rules", "total_cost"] - pim.loc["Model policy alone", "total_cost"]
v1, v2, v3 = (rows_v.loc[rows_v.model_version == k, "test_pr_auc"].iloc[0] for k in ("20261020-153708-3c73d465", "20261020-154750-a6506be0", "20261026-184402-d37e9778"))
viva = [
    ("Why PR-AUC and not accuracy?", f"Only {100 * fr:.2f}% of test transactions are fraud, so 'approve everything' is {100 * (1 - fr):.2f}% accurate and catches nothing. PR-AUC looks only at how well the rare class is ranked and how many alerts are real. Test PR-AUC is {test['pr_auc']:.4f}.", "artifacts/metrics.json split.test.fraud_rate, test.pr_auc; Saito and Rehmsmeier 2015"),
    ("Why a time-based split?", "Fraud is detected in the future from the past. Fit = oldest 85% of the training file, validation = last 15% by time, test = the later file; the test set is used once, after calibration and the threshold are fixed.", "fraudlens/fraudkit/train.py:349-358; tables/dataset_splits.csv"),
    ("How do you stop data leakage in the features?", "Every card feature is computed from strictly earlier rows of the same card (rows sorted by card, time, position). Merchant encoding is out-of-fold and fitted on the fit split; merchant/category risk ignores labels younger than 7 days. Tests truncate the future and check that features do not change.", "fraudkit/features.py:147-165, 92-99, 128-135, 311-320; tests/test_features.py"),
    ("What is target encoding and why out-of-fold?", "Merchant is replaced by its smoothed fraud rate. If a row's own label were used, the feature would leak the answer, so fit rows are encoded with a mapping learned on the other 4 folds.", "fraudkit/features.py:284-320"),
    ("Why LightGBM?", f"On validation PR-AUC the tuned LightGBM scored {lb['LightGBM tuned (Optuna)']:.4f} against {lb['Random forest']:.4f} (random forest), {lb['LightGBM default']:.4f} (default LightGBM) and {lb['Logistic regression']:.4f} (logistic regression). Caveat: only LightGBM was tuned, so part of the gap is tuning, not the algorithm.", "tables/leaderboard_final.csv; fraudkit/train.py:397-447"),
    ("How was the model tuned?", f"Optuna TPE sampler, {M['tuning']['trials']} trials, 3-fold stratified CV on PR-AUC, with 25% of genuine rows kept inside each fold for speed; best CV PR-AUC {M['tuning']['best_cv_pr_auc']:.4f}. The final model is trained on the same 25% genuine subsample plus all frauds.", "fraudkit/train.py:453-477, 516-520; tables/optuna_trials_final.csv"),
    ("Why calibrate, and which method won?", f"Training on a down-sampled genuine class and weighting frauds shifts the scores away from true probabilities. Isotonic, Platt and beta were compared by out-of-sample log-loss on two time halves of validation: isotonic {cal.loc['isotonic', 'log_loss']:.5f}, Platt {cal.loc['platt', 'log_loss']:.5f}, beta {cal.loc['beta', 'log_loss']:.5f}. Platt won.", "metrics.json calibration.comparison; fraudkit/score.py:42-85"),
    ("Why did the app first show only 0% or 100%?", f"Isotonic regression is a step function: on the validation halves it produced only {int(cal.loc['isotonic', 'distinct_values'])} distinct values and {100 * cal.loc['isotonic', 'share_at_floor']:.2f}% of rows sat at the floor. Platt keeps the ranking and gives {test['distinct_probabilities']:,} distinct values on the test set (after rounding to 12 decimals).", "metrics.json calibration.comparison, test.distinct_probabilities"),
    ("What is the 0-999 risk score?", f"A log-odds scale: p = 1e-6 maps to 0 and p = 0.999 to 999; about {X['risk_score']['points_to_double_odds']:.1f} points double the odds. The decision threshold {M['threshold']['value']:.4f} is score {X['risk_score']['score_at_decision_threshold']}. Decisions use the probability; the score is for reading.", "fraudkit/score.py:29-39; _data/extra_facts.json"),
    ("How is the decision threshold chosen?", f"On validation, by minimising total cost = value of missed frauds + EUR {M['threshold']['review_cost']:g} per flagged transaction. That gave threshold {M['threshold']['value']:.4f}.", "metrics.json threshold; fraudkit/costs.py"),
    ("Flat threshold or expected loss, and which is served?", f"Expected-loss policy flags when p x amount >= k x review cost (k = {M['decision']['expected_loss']['k']:g}). On validation the flat threshold was cheaper ({M['decision']['threshold']['validation']['total_cost']:,.2f} vs {M['decision']['expected_loss']['validation']['total_cost']:,.2f}), so it is served, even though expected loss was cheaper on test ({M['decision']['expected_loss']['test']['total_cost']:,.2f} vs {M['decision']['threshold']['test']['total_cost']:,.2f}); switching on test would be tuning on test. k = 8 is the largest value in the grid.", "tables/decision_policy_comparison.csv; fraudkit/decision.py"),
    ("What do the policy rules do and do they help?", f"They only raise the band (e.g. EUR 12,000 at 2,500 km, which the model scored Low). On the simulated test set they add {extra_rev} reviews, catch no extra fraud and raise cost by EUR {extra_cost:,.0f}; their value is protection outside the training data.", "tables/policy_rules_impact_test.csv; fraudkit/rules.py:87-111; screenshot 2c"),
    ("How do the reason codes work?", "SHAP TreeExplainer gives each feature's contribution in log-odds; related features are grouped; contributions of at least 0.05 become red (top 4) or green (top 2) reasons filled with the transaction's real values. SHAP + base value reconstruct the raw model output (tested to 1e-6).", "fraudkit/explain.py:347-364; tests/test_explain.py; _data/shap_examples.json"),
    ("What is PSI and what did drift monitoring find?", f"PSI = sum((a - e) ln(a / e)) over bins; < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant. It showed the lifetime counter card_txn_count at PSI {S['card_txn_count_psi_test_vs_fit']:.2f} (replaced by n_txn_30d) and, in 2020-12, four significant features led by n_txn_7d (PSI 1.030).", "fraudkit/monitoring.py:39-50; tables/drift_*.csv"),
    ("How did the model improve across runs and why?", f"Test PR-AUC {v1:.4f} -> {v2:.4f} (drifting counter replaced by n_txn_30d) -> {v3:.4f} (7 research-based features, Platt calibration). Between v2 and v3 the validation-chosen threshold moved 0.0238 -> 0.0837.", "tables/runs_comparison.csv; tables/feature_ablation.csv"),
    ("What does the 95% bootstrap CI say?", f"Test PR-AUC {test['pr_auc']:.4f}, 95% CI {test['pr_auc_ci95'][0]:.4f}-{test['pr_auc_ci95'][1]:.4f} from {test['bootstrap_resamples']} resamples of test rows. It treats rows as independent although a card's transactions are correlated, so it is optimistic.", "metrics.json test; fraudkit/train.py:215-229"),
    ("What are the limits of your evaluation?", "Simulated data; one split; 908 of the 924 test cards also appear in training (no new-card test); Optuna CV is shuffled, not time/group aware; labels arrive instantly; the money metric assumes every flagged fraud is fully recovered and 'no model' means zero protection.", "_data/extra_facts.json; REPORT_FACTS.md section 8"),
    ("What was wrong with the base paper's evaluation?", "Per the team's review: a tiny balanced test set (~124 rows), preprocessing (under-sampling, correlation-based feature selection, outlier removal) before the split, and 'ROC-AUC' that matches balanced accuracy. Not verified against the paywalled paper.", "docs/Literature_Review_Fraud_Detection.docx section 2; INCONSISTENCIES I11"),
    ("How is the app secured?", "PBKDF2-HMAC-SHA256 (600,000 iterations) with lockout after 5 failures for 15 minutes, 60-minute session timeout, server-side roles, hashed API keys, rate limit 120 requests per minute per key, HTML escaping and CSV-injection protection. SECURITY.md lists what is still missing for real card data.", "fraudkit/auth.py:35,60-72; configs/default.yaml; SECURITY.md"),
    ("Which roles exist and what can each do?", "viewer: read-only pages; analyst: also score, work cases, feed the queue; admin: also clear the queue, retrain, manage users and API keys, view the audit log.", "tables/role_permission_matrix.csv; figures/role_matrix.png"),
    ("How does the API score a transaction?", "Check key, rate limit and batch size; validate fields; build features from the card's earlier transactions; LightGBM raw score; Platt probability; risk score and band; SHAP reasons; policy rules; decision and expected loss; JSON response with the model version.", "figures/scoring_flow.png; api/main.py:270-327"),
    ("How is data stored?", "SQLite in WAL mode with versioned migrations: users, api_keys, admin_audit, cases, audit, case_comments, watchlist. Card numbers are stored as the last four digits; blocked cards as a SHA-256 hash.", "figures/sqlite_er.png; fraudkit/db.py:14-107; fraudkit/security.py:30-40"),
    ("How was it tested?", "81 pytest tests (80 pass; the one failure needs network access to Kaggle), ruff lint and format checks, and a GitHub Actions workflow running ruff and pytest; tests needing the dataset or a trained model skip themselves in CI.", "_data/pytest_full_run.txt; .github/workflows/fraudlens-ci.yml"),
    ("Is training reproducible?", "Fixed seeds and deterministic LightGBM: three archived runs share the model-hash suffix a6506be0 with identical test metrics, and tests/test_train_repro.py compares two fast runs to 10 decimals. Wall-clock times differ (502-1,235 s) with machine load.", "tables/runs_comparison.csv; tests/test_train_repro.py"),
    ("What is missing before real use?", "SSO/MFA, TLS, a server database, shared rate limiting, a secrets manager, PCI DSS scoping, model governance, trust boundary for the pickle model file, real merchant/device/IP data, delayed labels, and a fairness review (not done). The hosted demo script is prepared but was not deployed.", "SECURITY.md; fraudlens/deploy/space; REPORT_FACTS.md section 12"),
]
assert len(viva) == 25
write("viva_questions.csv", [{"no": i, "question": q, "short_answer": a, "reference": r} for i, (q, a, r) in enumerate(viva, 1)])
