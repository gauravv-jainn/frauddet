"""REPORT_FACTS.md sections 1-7."""

from __future__ import annotations

import pandas as pd
import yaml
from facts_common import AT, FL, M, S, SHAP, TEST, X, card_overlap, docstring_first_line, ieee_authors, md_table, n, pct, src, tab

SP = M["split"]
CAL = {r["method"]: r for r in M["calibration"]["comparison"]}


def F4(v): return f"{v:.4f}"
def F5(v): return f"{v:.5f}"
def F6(v): return f"{v:.6f}"
def F2(v): return f"{v:.2f}"
def G6(v): return f"{v:.6g}"
def N0(v): return n(v)
def N2(v): return n(v, 2)
def PCT2(v): return pct(v, 2)
def PCT3(v): return pct(v, 3)


FMT_SPL = {"rows": N0, "frauds": N0, "fraud_rate": PCT3}
FMT_CAT = {"rows": N0, "frauds": N0, "fraud_rate": PCT2}
FMT_BP = {"value": G6}
FMT_LB = {"pr_auc": F4, "roc_auc": F4, "brier": F5}
FMT_CAL = {"log_loss": F5, "brier": F6, "share_at_floor": PCT2}
FMT_DEC = {"precision": F4, "recall": F4, "total_cost": N2, "saved_vs_no_model": N2}
FMT_PR = {"precision": F4, "recall": F4, "false_alarms_per_10k_genuine": F2, "review_cost_total": N0, "total_cost": N0}


def s1() -> str:
    abstract = (
        f"Credit card fraud is rare ({pct(SP['test']['fraud_rate'], 2)} of the {n(SP['test']['rows'])} test transactions here) and costly, so accuracy is a "
        "misleading measure and every alert must be justified to a human analyst. This project reproduces and critiques the evaluation of Tanouz et al. (2021) "
        f"and then builds FraudLens, an explainable fraud-scoring system. A leakage-free per-card feature pipeline ({len(M['features'])} features) feeds a LightGBM model tuned "
        "with Optuna on PR-AUC and trained on a time-ordered split of the Kaggle Sparkov simulated transactions. Platt-scaled probabilities, chosen by out-of-sample log-loss, "
        "replace an isotonic calibrator that produced scores of only 0% or 100%, and a 0-999 log-odds risk score makes them readable. A money-aware threshold, configurable "
        "policy rules for out-of-range cases and SHAP-based plain-English reasons turn scores into approve, step-up, review or block decisions. The system is delivered as a "
        "Streamlit analyst application with role-based access, a SQLite case queue with audit trail, PSI drift monitoring and a FastAPI scoring service. On the untouched test "
        f"period the model reaches PR-AUC {TEST['pr_auc']:.4f} (95% CI {TEST['pr_auc_ci95'][0]:.4f}-{TEST['pr_auc_ci95'][1]:.4f}), {pct(AT['recall'])} recall and "
        f"{pct(AT['precision'])} precision at {AT['false_alarms_per_10k_genuine']:.1f} false alarms per 10,000 genuine transactions. All results come from simulated data and a single split."
    )
    words = len(abstract.split())
    assert words <= 200, words
    stack = tab("tech_stack.csv")
    return f"""## 1. Title options, abstract, keywords, tech stack

### 1.1 Title options (suggestions written for this kit, not facts)
1. FraudLens: An Explainable, Calibrated Credit Card Fraud Detection System with an Analyst Workflow
2. Explainable Credit Card Fraud Scoring with Calibrated Risk Scores and Cost-Aware Decisions
3. From Paper to Product: Re-evaluating and Extending Machine-Learning Credit Card Fraud Detection
4. Credit Card Fraud Detection Using LightGBM, SHAP and a Case-Management Application

### 1.2 Abstract draft ({words} words; every number is from `fraudlens/artifacts/metrics.json`)
{abstract}

### 1.3 Keywords
credit card fraud; class imbalance; LightGBM; probability calibration; SHAP explanations; precision-recall; population stability index (concept drift); decision support.

### 1.4 Tech stack with exact versions
{md_table(stack, ['category', 'component', 'version', 'source'], ['Category', 'Component', 'Version', 'Source'])}

{src('`fraudlens/artifacts/metrics.json` -> versions, machine', '`fraudlens/requirements*.txt`', 'commands noted in the table')}
"""


def s2() -> str:
    return """## 2. Problem, motivation, objectives, scope, out-of-scope

*Sources: `fraudlens/README.md` (What it does, Limitations), `fraudlens/SECURITY.md`, `fraudlens/configs/default.yaml`, `docs/Literature_Review_Fraud_Detection.docx`.*

**Problem.** Fraudulent card transactions are a tiny share of all transactions, so a model that approves everything looks accurate yet catches nothing; the evaluation protocol (split, metric, uncertainty) matters as much as the algorithm (team's literature review, sections 1 and 5). Analysts can only review a limited number of alerts, so every alert needs a reason and a cost justification.

**Motivation.**
- The base paper (Tanouz et al., 2021) reports very high scores on a tiny balanced test set; the team wanted to re-test that with a realistic protocol (`docs/Literature_Review_Fraud_Detection.docx` section 2).
- A model score alone is not a product: analysts need explanations, a workflow, access control and monitoring.

**Objectives (measurable).**
1. Evaluate the base paper's pipeline against a protocol with a time-ordered split, PR-AUC and a bootstrap confidence interval.
2. Build a feature pipeline in which a row never sees later rows of the same card (verified by tests).
3. Train and compare four models on a time split and report validation and test PR-AUC with a 95% CI.
4. Produce calibrated probabilities and a 0-999 risk score that are not stuck at 0% or 100%.
5. Choose the review decision by money (missed fraud + review cost) on validation, not on test.
6. Explain every flagged transaction in plain English with SHAP-based reason codes.
7. Provide an analyst application (case queue, audit trail, roles, drift monitoring) and a scoring API.
8. Verify the system with automated tests and a CI workflow.

**Scope (what exists in the repo).** Notebook reproducing the paper (`CreditCard_Fraud_MiniProject.ipynb`); the `fraudlens/` package (`fraudkit/`), a 10-page Streamlit app, a FastAPI service, 81 tests, Docker files, a Hugging Face Space deployment script, and a viva report PDF.

**Out of scope / not done (each is stated in the repo or checked by search).**
- Real card data, merchant/device/IP signals, chargeback delays (README Limitations).
- SSO/MFA, TLS, server database, shared rate limiter, secrets manager, PCI DSS review, model-governance approval step (`fraudlens/SECURITY.md` "Required before production").
- Automated retraining and alerting (the Monitoring page shows drift but nothing pages anyone).
- Fairness / bias analysis: NOT FOUND (a search for fairness, disparate, bias in the repo's code and docs returned nothing).
- Deep-learning or graph models; real-time streaming infrastructure (not present in the repo).
- A live hosted deployment: the Hugging Face Space script exists (`fraudlens/scripts/deploy_space.py`) but was NOT run.
"""


def s3() -> str:
    bp = tab("base_paper_vs_ours.csv")
    ref = tab("references_verified_final.csv").set_index("key").loc["tanouz2021"]
    return f"""## 3. Base paper (Tanouz et al.): summary, weaknesses, numbers vs ours

**Citation (metadata VERIFIED in Crossref, DOI {ref['doi']}):** {ieee_authors(ref['authors'])}, "{ref['registry_title']}," {ref['container']}, {ref['year']}, pp. {ref['pages']}.

**What the paper says (abstract, via Semantic Scholar):** card fraud is growing; the work proposes machine-learning classifiers such as logistic regression, random forest and Naive Bayes for a heavily imbalanced dataset and computes accuracy, precision, recall, F1, confusion matrix and ROC-AUC.

**Pipeline as summarised by the team's review** (`docs/Literature_Review_Fraud_Detection.docx` section 2; the full paper is behind an IEEE paywall, so this is UNVERIFIED against the paper): compare Decision Tree, Random Forest, Logistic Regression and Naive Bayes on the ULB dataset; normalise Time and Amount; under-sample the genuine class to 1:1; drop features correlated with the label; remove outliers; only then split train/test. Reported best model: Random Forest with 96.77% accuracy, 100% precision, 91.11% recall (UNVERIFIED against the paper).

**Weaknesses found by the team** (same document; the team's analysis, not independently re-run in this kit):
1. The test set is balanced and tiny: the reported metrics are consistent with about 124 test rows, about 45 of them fraud, while the real fraud rate is 0.17%.
2. Preprocessing (under-sampling, label-based feature selection, outlier removal) happens before the split, which is data leakage.
3. The reported "ROC-AUC" matches balanced accuracy within about 0.01, as if computed from hard 0/1 predictions.
4. Differences between models are a few transactions, with no confidence intervals.

### Paper numbers vs reproduced numbers vs FraudLens
{md_table(bp, ['setting', 'dataset', 'test_rows', 'precision_pct', 'recall_pct', 'false_alarms', 'source'], ['Setting', 'Dataset', 'Test rows', 'Precision %', 'Recall %', 'False alarms', 'Source'])}

**Read this before using the table.**
- Rows 1-4 are copied from `docs/Literature_Review_Fraud_Detection.docx` Table 1. `CreditCard_Fraud_MiniProject.ipynb` has **no saved cell outputs** (0 of its code cells have outputs), so these numbers could not be regenerated from an artifact here: **NOT re-verified**. Re-run the notebook in Colab (and keep the outputs) before printing them in the report.
- Row 5 uses a different dataset (Sparkov simulated, {n(SP['test']['rows'])} test rows) and a different protocol, so it must not be presented as "better than the paper".
- The paper's own numbers could not be checked (paywall): UNVERIFIED.
"""


def s4() -> str:
    lit = tab("literature_survey.csv")
    return f"""## 4. Literature survey (every reference verified against a registry)

Verification method (`report_kit/scripts/verify_refs.py`, `verify_refs2.py`): each candidate DOI was looked up in Crossref and kept **only if** the registry record matched the title (similarity >= 0.90), first author and year; arXiv records through the arXiv API; web pages by fetching them. **No DOI in this kit was written from memory without registry confirmation.** Where an abstract was not available, the "method / dataset / result / limitation" cells say NOT STATED instead of guessing. Full bibliographic entries: `references.md`; machine-readable: `tables/literature_survey.csv`, `tables/references_verified_final.csv`.

{md_table(lit, ['no', 'authors', 'year', 'venue', 'method', 'dataset', 'result', 'limitation', 'verification'], ['#', 'Authors', 'Year', 'Venue', 'Method', 'Dataset', 'Result', 'Limitation', 'Verification'])}

Content columns come from the abstract (Semantic Scholar / arXiv) or from the team's own review (`docs/Literature_Review_Fraud_Detection.docx`), as named in `tables/literature_survey.csv` -> `summary_source`. **UNVERIFIED items:** the base paper's reported numbers (paywall); Platt (1999/2000, year conflicts between sources; no DOI); see `references.md`.
"""


def s5() -> str:
    OV = card_overlap()
    cols = tab("dataset_columns_raw.csv")
    meaning = {
        "Unnamed: 0": ("row index of the CSV (dropped by the loader)", "dropped"), "trans_date_trans_time": ("transaction timestamp", "yes (time features)"),
        "cc_num": ("synthetic card number (card identity for history features)", "yes (grouping key only)"), "merchant": ("merchant name (the prefix 'fraud_' is stripped by the loader)", "yes (merchant_te)"),
        "category": ("merchant category", "yes"), "amt": ("transaction amount", "yes"), "first": ("cardholder first name", "dropped (personal data)"), "last": ("cardholder last name", "dropped (personal data)"),
        "gender": ("cardholder gender", "yes"), "street": ("cardholder street", "dropped (personal data)"), "city": ("cardholder home city", "context only"), "state": ("cardholder home state", "context only"),
        "zip": ("cardholder zip", "dropped"), "lat": ("home latitude", "yes (dist_home_km)"), "long": ("home longitude", "yes (dist_home_km)"), "city_pop": ("population of the home city", "yes (log_city_pop)"),
        "job": ("cardholder job title", "context only"), "dob": ("date of birth", "yes (age)"), "trans_num": ("transaction id", "key only"), "unix_time": ("timestamp as unix seconds", "dropped"),
        "merch_lat": ("merchant latitude", "yes (distance, speed)"), "merch_long": ("merchant longitude", "yes (distance, speed)"), "is_fraud": ("label: 1 = fraud", "target"),
    }
    cols["meaning (inferred from name and fraudkit/data.py)"] = cols["column"].map(lambda c: meaning.get(c, ("NOT FOUND", ""))[0])
    cols["used by FraudLens"] = cols["column"].map(lambda c: meaning.get(c, ("", "NOT FOUND"))[1])
    spl = tab("dataset_splits.csv")
    cat = tab("fraud_rate_by_category_train.csv")
    rr = X["raw_rows"]
    return f"""## 5. Dataset

- **Source:** {M['data']['source']}; loaded with `kagglehub` (`fraudlens/fraudkit/data.py:KAGGLE_DATASET`). Page: https://www.kaggle.com/datasets/kartik2112/fraud-detection (title verified; reachable). The generator is the Sparkov data generation tool (https://github.com/namebrandon/Sparkov_Data_Generation, reachable). **Licence: NOT FOUND** (not checked on the Kaggle page).
- **Files and rows (counted from the cached CSVs):** `fraudTrain.csv` {n(rr['fraudTrain.csv'])} rows, `fraudTest.csv` {n(rr['fraudTest.csv'])} rows. The provided split is by time and is kept.
- **Fraud rate:** train file {pct(M['data']['train']['fraud_rate'], 3)} ({n(M['data']['train']['frauds'])} frauds in {n(M['data']['train']['rows'])} rows, {n(M['data']['train']['cards'])} cards); test file {pct(M['data']['test']['fraud_rate'], 3)} ({n(M['data']['test']['frauds'])} frauds in {n(M['data']['test']['rows'])} rows, {n(M['data']['test']['cards'])} cards).
- **Period:** {M['data']['train']['start']} to {M['data']['train']['end']} (train file); {M['data']['test']['start']} to {M['data']['test']['end']} (test file).
- **Currency:** none in the data; the app shows amounts as EUR (`fraudlens/configs/default.yaml` comment, Model report data card).
- **Personal data:** `clean()` drops first, last, street, zip and unix_time on load (`fraudlens/fraudkit/data.py`, `clean`); the data are synthetic.

### 5.1 Split sizes (what the model actually uses)
{md_table(spl, ['part', 'rows', 'frauds', 'fraud_rate', 'start', 'end'], ['Part', 'Rows', 'Frauds', 'Fraud rate', 'Start', 'End'], FMT_SPL)}

{src('`fraudlens/artifacts/metrics.json` -> split, data', '`tables/dataset_splits.csv`')}
The validation period is the last 15% of the training file by time (`configs/default.yaml` data.validation_fraction; `fraudkit/train.py:349-358`).

### 5.2 Columns of the raw files
{md_table(cols, ['column', 'pandas_dtype_in_csv', 'meaning (inferred from name and fraudkit/data.py)', 'used by FraudLens', 'nulls_in_train', 'distinct_in_train'], ['Column', 'dtype in CSV', 'Meaning (inferred; no official data dictionary was found)', 'Used by FraudLens', 'Nulls (train)', 'Distinct (train)'])}

{src('command: `report_kit/scripts/extra_facts.py` reading the cached `fraudTrain.csv`', '`tables/dataset_columns_raw.csv`')}

### 5.3 Facts about the training file that explain model behaviour
- Fraud rate in night hours (22:00-03:59) {pct(X['fraud_rate_train_night_hours_22_to_03'], 2)} vs {pct(X['fraud_rate_train_other_hours'], 2)} in other hours.
- Largest fraud amount {n(X['fraud_amt_train']['max'], 2)} (median {n(X['fraud_amt_train']['median'], 2)}); largest genuine amount {n(X['genuine_amt_train']['max'], 2)}; amounts above 2,000: {X['amt_over_2000_train']['rows']} rows, {X['amt_over_2000_train']['frauds']} frauds. A tree model cannot extrapolate beyond this, which motivated the policy rules.
- {OV[2]} of the {OV[1]} test cards also appear in the training file ({OV[0]} cards), so the test does not measure performance on unseen cardholders (command: `card_overlap()` in `report_kit/scripts/facts_common.py`, reading the cached CSVs).

Fraud rate by category (training file):

{md_table(cat.head(14), ['category', 'rows', 'frauds', 'fraud_rate'], ['Category', 'Rows', 'Frauds', 'Fraud rate'], FMT_CAT)}
"""


def s6() -> str:
    files = [f for f in sorted((FL / "fraudkit").glob("*.py")) if f.name != "__init__.py"] + [FL / "api" / "main.py"]
    mods = pd.DataFrame([{"file": f"fraudlens/{f.relative_to(FL).as_posix()}", "lines": len(f.read_text(encoding='utf-8').splitlines()), "purpose": docstring_first_line(f)} for f in files])
    pages = tab("pages_roles.csv")
    eps = tab("api_endpoints.csv")
    perm = tab("role_permission_matrix.csv")
    cfg = yaml.safe_load((FL / "configs" / "default.yaml").read_text(encoding="utf-8"))
    flat = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat.append({"key": f"{k}.{k2}", "value": v2})
        else:
            flat.append({"key": k, "value": v})
    return f"""## 6. Architecture

Figures: `figures/architecture.png` (system), `figures/scoring_flow.png` (one API request), `figures/sqlite_er.png` (database), `figures/training_pipeline_time_split.png`, `figures/role_matrix.png`. Sources of the diagrams are in `figures/diagrams_src/` (Graphviz source files with the extension dot, and `make_matplotlib_figures.py`).

### 6.1 Python modules (`fraudkit/` plus the API)
{md_table(mods, ['file', 'lines', 'purpose'], ['File', 'Lines', 'Purpose (first line of the module docstring)'])}

{src('command: `ast.get_docstring` and line count over the files listed (`report_kit/scripts/facts_part1.py`)')}

### 6.2 Streamlit pages (`fraudlens/app/common.py` PAGES; entry point `app/streamlit_app.py`)
{md_table(pages, ['section', 'page', 'file', 'permission_to_open', 'roles_that_can_open'], ['Section', 'Page', 'File', 'Permission', 'Roles that can open it'])}

Inside a page, extra rules apply (e.g. Check a transaction: typed input and what-if need `score`; Case queue: acting needs `decide_cases`, clearing needs `clear_queue`; Training Lab: Retrain needs `retrain`). Source: `fraudlens/app/views/*.py` (`allowed(...)` calls).

### 6.3 API endpoints (`fraudlens/api/main.py`)
{md_table(eps, ['method', 'path', 'auth', 'purpose', 'source'], ['Method', 'Path', 'Auth', 'Purpose', 'Source'])}

Limits: {cfg['api']['max_batch']} transactions per request (HTTP 413 above), {cfg['api']['rate_limit_per_minute']} requests per minute per key (HTTP 429), field validation errors return HTTP 422; every response carries `request_id` and `model_version`. Source: `configs/default.yaml` api; `api/main.py:270-327`; `tests/test_api.py` (8 tests).

### 6.4 SQLite schema (`fraudlens/fraudkit/db.py` MIGRATIONS, schema version 3)
Tables: `cases`, `audit`, `users`, `api_keys`, `admin_audit`, `watchlist`, `case_comments` (columns and foreign keys in `figures/sqlite_er.png`; DDL in `fraudkit/db.py:14-107`). WAL journal mode and `busy_timeout` are set in `connect()` (`db.py:114-120`). Migrations are applied in one transaction each and tracked with `PRAGMA user_version` (`db.py:123-135`); `tests/test_platform.py::test_migration_upgrades_old_case_database` upgrades an old database in place.

### 6.5 Authentication and roles (`fraudlens/fraudkit/auth.py`)
{md_table(perm, ['permission', 'viewer', 'analyst', 'admin'], ['Permission', 'viewer', 'analyst', 'admin'])}

Passwords: PBKDF2-HMAC-SHA256, 600,000 iterations, random 16-byte salt, minimum length 10 with a letter and a digit, not containing the username (`auth.py:35-36,49-72`). Lockout after {cfg['app']['max_failed_logins']} failed attempts for {cfg['app']['lockout_minutes']} minutes and a {cfg['app']['session_timeout_minutes']}-minute session timeout (`configs/default.yaml` app). API keys: random `fl_...` token shown once, stored as a SHA-256 hash, revocable, with use counts (`auth.py:248-268`). The OWASP Password Storage Cheat Sheet lists "PBKDF2-HMAC-SHA256: 600,000 iterations" (page fetched during this kit's verification).

### 6.6 Configuration (`fraudlens/configs/default.yaml`, validated by pydantic in `fraudkit/config.py`)
{md_table(pd.DataFrame(flat), ['key', 'value'], ['Key', 'Value'])}

### 6.7 Deployment
- **Local Docker:** `fraudlens/Dockerfile` (python:3.11-slim, non-root user, health check) and `fraudlens/docker-compose.yml` (api on 127.0.0.1:8000, app on 127.0.0.1:8501, artifacts mounted read-only, data in a named volume).
- **Hugging Face Space (prepared, NOT deployed):** `fraudlens/deploy/space/Dockerfile`, `fraudlens/fraudkit/bootstrap.py` (refuses to start without admin and demo passwords, downloads artifacts, creates accounts) and `fraudlens/scripts/deploy_space.py`. The Space image was built and run locally during development and served sign-in, the dashboard and a 500-row batch (`fraudlens/PROGRESS.md` M12); no public URL exists yet.
- **CI:** `.github/workflows/fraudlens-ci.yml` (section 9).
"""


def s7() -> str:
    feats = tab("features_catalog.csv")
    lb = tab("leaderboard_final.csv")
    best = M["tuning"]["best_params"]
    bp = pd.DataFrame([{"parameter": k, "value": v} for k, v in best.items()])
    cal = pd.DataFrame(M["calibration"]["comparison"])
    ex = []
    for e in SHAP:
        lines = [f"**{e['case']}** (`trans_num` {e['trans_num']}; dataset label is_fraud = {e['row_label_is_fraud']}; amount {e['amount']:.2f}; category {e['category']}; hour {e['hour']})",
                 f"- calibrated probability {e['recomputed_p']:.6f} (stored value {e['stored_p']:.6f}); risk score {e['risk_score']}; band {e['final_band']}; decision **{e['decision']}**; expected loss {e['expected_loss']:.2f}",
                 f"- summary text shown to the analyst: \"{e['summary']}\""]
        lines += [f"- raises risk (+{r['shap']:.2f} log-odds): {r['text']}" for r in e["raising"]]
        lines += [f"- lowers risk ({r['shap']:.2f} log-odds): {r['text']}" for r in e["lowering"]]
        lines.append(f"- policy rules fired: {len(e['rules'])}")
        ex.append("\n".join(lines))
    thr = M["threshold"]
    rules = X  # noqa: F841
    ra = M["rules_auto"]
    return f"""## 7. Methodology

### 7.1 Features ({len(M['features'])} model inputs) and leakage prevention
Rule (docstring of `fraudkit/features.py:1-7`): every card-behaviour feature of a transaction is computed **only from strictly earlier transactions of the same card**. Rows are sorted by (card, time, original position) so ties are resolved deterministically (`features.py:153-155`); the rolling-window helper counts rows `j < i` with `t_j >= t_i - window` (`features.py:92-99`). Merchant and category risk use labels only if they are at least 7 days old (`features.py:17,128-135`). Merchant target encoding is out-of-fold and fitted on the fit split only (`features.py:284-320`).
Tests (`fraudlens/tests/test_features.py`, 7 tests; names copied from the file): `test_no_feature_uses_future_rows`, `test_future_perturbation_does_not_change_past`, `test_featurize_with_history_matches_batch`, `test_velocity_and_first_time_semantics`, `test_haversine_known_distance`, `test_merchant_encoder_oof_does_not_use_own_label`, `test_delayed_risk_ignores_recent_labels` (run results in section 9).

{md_table(feats, ['feature', 'group', 'definition', 'source', 'mean_abs_shap', 'importance_rank'], ['Feature', 'Group', 'Definition / formula', 'Code', 'Mean abs SHAP (log-odds)', 'Rank'])}

{src('`fraudlens/fraudkit/features.py` (line numbers in the Code column)', '`fraudlens/artifacts/metrics.json` -> global_importance (5,000 random validation rows)', '`tables/features_catalog.csv`')}
Context columns (not model inputs): prev_city_km, minutes_since_prev, card_mean_amt, cat_mean_amt, card_txn_count, card_max_amt, amt_to_card_max, cat_share, hour_share (`features.py:68-78`). `card_txn_count` was removed from the model because its PSI was {S['card_txn_count_psi_test_vs_fit']:.2f} on the test period (section 8.4).

### 7.2 Models, tuning, final training
- Baselines trained on all fit rows (`fraudkit/train.py:397-447`): logistic regression (log1p + median imputation + standard scaling + one-hot, `C=1.0`, `max_iter=2000`); random forest (200 trees, `min_samples_leaf=5`, `max_features="sqrt"`, `max_samples=0.3`); LightGBM with default parameters. **None of the baselines is tuned**, only LightGBM is.
- Tuned LightGBM: Optuna TPE sampler (seed 42), {M['tuning']['trials']} trials, {M['tuning']['cv_folds']}-fold stratified shuffled CV on the fit period, objective mean PR-AUC, 25% of genuine rows kept per fold (`train.py:453-477`). Search ranges: n_estimators 150-1200 (step 50), learning_rate 0.01-0.2 (log), num_leaves 15-255 (log), min_child_samples 10-300 (log), subsample 0.5-1.0, colsample_bytree 0.4-1.0, reg_lambda and reg_alpha 1e-3-10 (log), scale_pos_weight 1-10 (log). Best CV PR-AUC {M['tuning']['best_cv_pr_auc']:.4f}.
- The final model is refit with the best parameters on the fit rows with the same 25% genuine subsample (`train.py:516-520`), then scored on validation.

Best parameters (`metrics.json` tuning.best_params):

{md_table(bp, ['parameter', 'value'], ['Parameter', 'Value'], FMT_BP)}

Validation leaderboard (last 15% of the training file, {n(SP['validation']['rows'])} rows, {n(SP['validation']['frauds'])} frauds):

{md_table(lb, ['model', 'pr_auc', 'roc_auc', 'brier', 'fit_seconds', 'selected'], ['Model', 'PR-AUC', 'ROC-AUC', 'Brier', 'Fit seconds', 'Selected'], FMT_LB)}

### 7.3 Calibration
Down-sampling genuine rows and weighting frauds shifts raw scores away from true probabilities, so a calibrator is fitted on the validation period. Three calibrators are compared on two time-halves of validation (fit on one half, score on the other, both ways); the lowest out-of-sample log-loss wins (`train.py` calibration stage, `score.py:42-85`). Output is clipped to [1e-12, 1-1e-12] so a probability is never exactly 0 or 1.

{md_table(cal, ['method', 'log_loss', 'brier', 'distinct_values', 'share_at_floor'], ['Method', 'Log-loss', 'Brier', 'Distinct values', 'Share at floor'], FMT_CAL)}

Chosen: **{M['calibration']['method']}**. Test Brier score {TEST['brier_before_calibration']:.5f} before and {TEST['brier_after_calibration']:.5f} after calibration; test log-loss {TEST['log_loss_calibrated']:.5f}; {n(TEST['distinct_probabilities'])} distinct probabilities after rounding to 12 decimals ({n(X['risk_score']['distinct_p_values_test'])} distinct raw values). Reliability table: `tables/reliability_test_final.csv`. Why scores once looked like only 0% or 100%: isotonic regression is a step function (the validation halves show {int(cal.set_index('method').loc['isotonic', 'distinct_values'])} distinct values and {pct(cal.set_index('method').loc['isotonic', 'share_at_floor'], 2)} at the floor above). The "95% of test rows exactly 0 and 94 distinct values" figures in `PROGRESS.md` M11 describe the retired isotonic model whose test scores were not kept: **NOT re-derivable from retained artifacts**.

### 7.4 Risk score 0-999
`risk_score(p)` = round((logit(p) - logit(1e-6)) / (logit(0.999) - logit(1e-6)) x 999), clipped to 0-999 (`score.py:29-39`). About {X['risk_score']['points_to_double_odds']:.1f} points double the odds. On the test set the median score is {X['risk_score']['genuine_median']:.0f} for genuine and {X['risk_score']['fraud_median']:.0f} for fraud transactions; the decision threshold p = {thr['value']:.4f} is score {X['risk_score']['score_at_decision_threshold']}, High (p = 0.30) is {X['risk_score']['score_at_high_0.30']} and Critical (p = 0.70) is {X['risk_score']['score_at_critical_0.70']} (command: `report_kit/scripts/extra_facts.py`). Decisions use the probability; the score is for reading.

### 7.5 Decision logic
- **Threshold chosen on validation by cost:** total cost = amount of frauds not flagged + EUR {thr['review_cost']:g} x flagged transactions; minimum at p = {thr['value']:.4f} (`metrics.json` threshold; `fraudkit/costs.py`).
- **Two policies compared on validation** (`fraudkit/decision.py`): flat threshold (flag when p >= t) vs expected loss (flag when p x amount >= k x review cost; k from the grid 0.25-8, here k = {M['decision']['expected_loss']['k']:g}, the largest grid value). The cheaper one on validation is served: **{M['decision']['served']}**.
- **Actions:** block if band is Critical (p >= 0.70 or a Critical rule), review if the policy flags it or a High rule fired, step-up if band is Medium or higher or a Medium rule fired, otherwise approve (`decision.py:43-52`).
- Bands: Medium starts at the decision threshold, High at 0.30, Critical at 0.70 (`configs/default.yaml` bands).

{md_table(tab('decision_policy_comparison.csv'), None, None, FMT_DEC)}

### 7.6 Policy rules (guardrail; can only raise the band)
Added after a EUR 12,000 purchase 2,500 km from home scored Low (0.01%): in the training data no fraud exceeds EUR {n(X['fraud_amt_train']['max'], 2)} and amounts above 2,000 are never fraud, so the tree cannot extrapolate (`fraudkit/rules.py:1-8`; screenshot `2c_policy_rules.png`). Limits are derived from training data (`metrics.json` rules_auto):
- large amount >= {n(ra['review_amount'])} -> High ({ra['derivation']['review_amount']});
- far from home >= {n(ra['far_from_home_km'])} km -> Medium ({ra['derivation']['far_from_home_km']});
- spending spike >= {ra['spend_spike_ratio']:g}x the card's average and amount >= {n(ra['spend_spike_min_amount'])} -> High ({ra['derivation']['spend_spike']});
- impossible travel (speed flag and a jump >= far-from-home limit) -> High; burst of >= 5 other transactions in 1 hour -> High; any key feature outside the training range -> Medium; blocked card -> Critical; watched merchant -> High (`rules.py:87-122`, `configs/default.yaml` rules).
Measured effect on the test set with the final model:

{md_table(tab('policy_rules_impact_test.csv'), ['system', 'flagged', 'tp', 'fp', 'fn', 'precision', 'recall', 'false_alarms_per_10k_genuine', 'review_cost_total', 'total_cost'], None, FMT_PR)}

{md_table(tab('policy_rules_per_rule_test.csv'), None, None)}

On this simulated test set the rules add reviews and catch no extra fraud; their purpose is protection outside the training data. Source: `tables/policy_rules_*.csv` (command: `fraudkit.rules.policy_impact`, run by `report_kit/scripts/extra_facts.py`).

### 7.7 SHAP reason codes: three real examples
SHAP `TreeExplainer` gives each feature's contribution in log-odds; related features are grouped; groups at or above 0.05 log-odds become reasons (top 4 raising, top 2 lowering by default); wording is filled with the transaction's real values (`fraudkit/explain.py:334-376`). SHAP values plus the base value reconstruct the model's raw margin (`tests/test_explain.py`, tolerance 1e-6). The three examples below were produced by calling `explain_frame` on rows of `artifacts/test_scored.parquet` with the served model and policy (`report_kit/scripts/extract_results.py`; raw output `_data/shap_examples.json`):

{chr(10).join(chr(10) + b for b in ex)}

### 7.8 Drift monitoring (PSI)
For every feature and the score, the reference profile stores 10 quantile bins of the fit period (`fraudkit/monitoring.py:53-76`); new data is binned with the same edges and PSI = sum((actual - expected) x ln(actual / expected)) with shares clipped at 1e-4. Reading: < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant (`monitoring.py:13-14,39-50`). Results are in section 8.4.
"""
