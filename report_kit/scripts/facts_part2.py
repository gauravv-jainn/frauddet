"""REPORT_FACTS.md sections 8-16."""

from __future__ import annotations

import subprocess

import pandas as pd
from facts_common import AT, FL, M, REPO, S, TEST, X, card_overlap, loc, md_table, n, pct, py_files, pytest_summary, src, tab

SP = M["split"]


def F4(v): return f"{v:.4f}"
def F5(v): return f"{v:.5f}"
def F2(v): return f"{v:.2f}"
def N0(v): return n(v)
def N1(v): return n(v, 1)
def PCT1(v): return pct(v, 1)
def DATE(v): return str(v)[:16].replace("T", " ")


RUNS = tab("runs_comparison.csv")
FMT_Q = {"started_utc": DATE, "finished_utc": DATE, "wall_clock_seconds": N1, "tuning_seconds": N1, "test_pr_auc": F4, "test_pr_auc_ci95_low": F4, "test_pr_auc_ci95_high": F4,
         "test_roc_auc": F4, "test_brier_after_calibration": F5, "precision": F4, "recall": F4, "false_alarms_per_10k_genuine": F2, "threshold": F4, "total_cost_eur": N0,
         "saved_vs_no_model_eur": N0, "caught_amount_eur": N0, "missed_amount_eur": N0, "review_cost_total_eur": N0, "flagged": N0}


def s8() -> str:
    OV = card_overlap()
    card_ov = f"{OV[2]} of the {OV[1]} test cards also appear in the training file ({OV[0]} cards),"
    r = RUNS.set_index("model_version")
    v1, v2, v3, v4, v5, v6 = (r.loc[k] for k in ["20261020-153708-3c73d465", "20261020-154750-a6506be0", "20261020-172652-a6506be0", "20261025-224238-a6506be0", "20261025-233115-3abe0ebd", "20261026-184402-d37e9778"])
    ab = tab("feature_ablation.csv")
    dm = tab("drift_test_period_by_month.csv")
    dov = tab("drift_test_period_overall.csv")
    d12 = tab("drift_2020_12_all_features.csv")
    cc = tab("drift_card_txn_count_by_month.csv")
    rel = tab("reliability_test_final.csv")
    lbr = tab("leaderboard_by_run.csv")
    lbp = lbr.pivot(index="model_version", columns="model", values="pr_auc").reset_index()
    cols_q = ["model_version", "n_features", "calibration", "test_pr_auc", "test_pr_auc_ci95_low", "test_pr_auc_ci95_high", "test_roc_auc", "test_brier_after_calibration", "precision", "recall", "false_alarms_per_10k_genuine"]
    cols_m = ["model_version", "wall_clock_seconds", "tuning_seconds", "threshold", "flagged", "tp", "fp", "fn", "caught_amount_eur", "missed_amount_eur", "review_cost_total_eur", "total_cost_eur", "saved_vs_no_model_eur"]
    dpr = v6.test_pr_auc - v1.test_pr_auc
    return f"""## 8. Results

All numbers below are read from `fraudlens/artifacts/metrics.json` (final run) and `fraudlens/artifacts/runs/*/metrics.json` (archive of the six runs); tables are also saved in `tables/`. **Evaluation protocol:** fit on the oldest 85% of the training file, tune with CV, calibrate and choose the threshold on the last 15% (validation), evaluate **once** on the later test file. Review cost EUR {M['threshold']['review_cost']:g} per flagged transaction.

### 8.1 Final model: test results (model `{M['run']['model_version']}`)
| Metric | Value | Source |
|---|---|---|
| Test transactions / frauds | {n(SP['test']['rows'])} / {n(SP['test']['frauds'])} | metrics.json split.test |
| PR-AUC (95% bootstrap CI, {TEST['bootstrap_resamples']} resamples) | {TEST['pr_auc']:.4f} ({TEST['pr_auc_ci95'][0]:.4f} - {TEST['pr_auc_ci95'][1]:.4f}) | test.pr_auc, pr_auc_ci95 |
| ROC-AUC | {TEST['roc_auc']:.4f} | test.roc_auc |
| Brier score before / after calibration | {TEST['brier_before_calibration']:.5f} / {TEST['brier_after_calibration']:.5f} | test.brier_* |
| Decision threshold (chosen on validation) | {M['threshold']['value']:.4f} | threshold.value |
| Flagged (review or block) | {n(AT['flagged'])} | test.at_threshold.flagged |
| True / false positives, false negatives | {n(AT['tp'])} / {n(AT['fp'])}, {n(AT['fn'])} | test.at_threshold |
| Precision / recall | {pct(AT['precision'], 2)} / {pct(AT['recall'], 2)} | test.at_threshold |
| False alarms per 10,000 genuine | {AT['false_alarms_per_10k_genuine']:.2f} | test.at_threshold |
| Fraud amount in the test period | {n(AT['fraud_amount'], 2)} | test.at_threshold.fraud_amount |
| Caught / missed amount | {n(AT['caught_amount'], 2)} / {n(AT['missed_amount'], 2)} | test.at_threshold |
| Review cost / total cost (missed + reviews) | {n(AT['review_cost_total'])} / {n(AT['total_cost'], 2)} | test.at_threshold |
| Saved vs "no model" | {n(AT['saved_vs_no_model'], 2)} | test.at_threshold.saved_vs_no_model |
| Training wall-clock of this run | {n(M['run']['total_seconds'], 1)} s (includes the computer sleeping; the identical-metrics archived run took {n(v5.wall_clock_seconds, 1)} s) | run.total_seconds; runs/20261025-233115-3abe0ebd |

The validation leaderboard, calibration comparison and decision-policy comparison are in section 7. The reliability table (observed vs predicted fraud rate in fixed probability bins) is `tables/reliability_test_final.csv` ({len(rel)} rows).

### 8.2 Every archived run (quality)
{md_table(RUNS, cols_q, ['Model version', 'Features', 'Calibration', 'Test PR-AUC', 'CI low', 'CI high', 'ROC-AUC', 'Brier (calibrated)', 'Precision', 'Recall', 'False alarms / 10k'], FMT_Q)}

### 8.3 Every archived run (money and time)
{md_table(RUNS, cols_m, ['Model version', 'Wall-clock s', 'Tuning s', 'Threshold', 'Flagged', 'TP', 'FP', 'FN', 'Caught EUR', 'Missed EUR', 'Review cost EUR', 'Total cost EUR', 'Saved vs no model EUR'], FMT_Q)}

{src('`tables/runs_comparison.csv` (generated by `report_kit/scripts/extract_results.py` from `fraudlens/artifacts/runs/*/metrics.json`)')}
Validation PR-AUC of each model in each run: `tables/leaderboard_by_run.csv`. Optuna history of the final run: `tables/optuna_trials_final.csv` (best CV PR-AUC {S['trials']['best_cv_pr_auc']:.4f}, median {S['trials']['median_cv_pr_auc']:.4f}, worst {S['trials']['worst_cv_pr_auc']:.4f}, first trial {S['trials']['first_trial_cv_pr_auc']:.4f}).

**Reading the run table.** Runs 2, 3 and 4 (`...154750`, `...172652`, `20261025-224238`) share the model-hash suffix `a6506be0` and identical test metrics: the same model was retrained to add stored artifacts (training ranges for the rules, decision data), with wall-clock times of {n(v2.wall_clock_seconds, 0)}, {n(v3.wall_clock_seconds, 0)} and {n(v4.wall_clock_seconds, 0)} s. Runs 5 and 6 have identical test metrics but different hash suffixes (INCONSISTENCIES I7).

### 8.4 Drift findings (PSI of the served model's inputs on the test period vs the fit-period reference)
Whole test period, top features (all stable, PSI < 0.10):

{md_table(dov.head(8), ['feature', 'kind', 'psi', 'status'], ['Feature', 'Kind', 'PSI', 'Status'], {'psi': F4})}

By month (top feature, number of moderate and significant features, PSI of the score):

{md_table(dm, ['month', 'rows', 'top_feature', 'top_psi', 'n_moderate', 'n_significant', 'score_psi'], ['Month', 'Rows', 'Top feature', 'Top PSI', 'Moderate', 'Significant', 'Score PSI'], {'rows': N0, 'top_psi': F4, 'score_psi': F4})}

December 2020, features above 0.10:

{md_table(d12[d12.psi >= 0.10], ['feature', 'psi', 'status'], ['Feature', 'PSI', 'Status'], {'psi': F4})}

- December has {n(dm.set_index('month').loc['2020-12', 'rows'])} test rows against {n(dm.set_index('month').loc['2020-11', 'rows'])} in November, so the velocity features (n_txn_7d, n_txn_24h, n_txn_30d, amt_sum_24h) shift. The model score barely moves (PSI {dm.set_index('month').loc['2020-12', 'score_psi']:.3f}). That December is a seasonal spike is the README's explanation; this kit did NOT test it.
- **The defect found by drift monitoring:** the lifetime counter `card_txn_count` had PSI {S['card_txn_count_psi_test_vs_fit']:.2f} on the whole test period vs the fit period (monthly values {cc.psi_card_txn_count.min():.2f} - {cc.psi_card_txn_count.max():.2f}, `tables/drift_card_txn_count_by_month.csv`) and was the 5th most important feature in the earlier model (PROGRESS M8, not re-derivable: that model is no longer stored). It was replaced by `n_txn_30d`.
- `n_txn_30d` itself shows moderate monthly drift (0.11 - 0.16 in July to November): see INCONSISTENCIES I4.

{src('`tables/drift_test_period_overall.csv`, `drift_test_period_by_month.csv`, `drift_2020_12_all_features.csv`, `drift_card_txn_count_by_month.csv`', 'command: `fraudkit.monitoring.drift_report` over `artifacts/test_scored.parquet` and `artifacts/reference_profile.json` (`report_kit/scripts/extract_results.py`, `extra_facts.py`)')}

### 8.5 Feature ablation (validation PR-AUC, two seeds)
Candidates were added one group at a time to the earlier model's feature set and trained with the tuned parameters; a group is kept only if it beats the baseline by more than twice the seed noise ({S['ablation_seed_noise']:.5f}). Run on model `{S['ablation_model_version']}` (the model before the research features).

{md_table(ab, ['candidate', 'val_pr_auc_seed42', 'val_pr_auc_seed7', 'mean', 'delta_vs_current_baseline', 'times_seed_noise'], ['Candidate', 'Seed 42', 'Seed 7', 'Mean', 'Delta vs baseline', 'x seed noise'], {'val_pr_auc_seed42': F5, 'val_pr_auc_seed7': F5, 'mean': F5, 'delta_vs_current_baseline': F5, 'times_seed_noise': F2})}

Kept by the rule (`artifacts/feature_ablation.json` -> kept): {', '.join(S['ablation_kept'])}. The model uses the simplest equal set (spending windows + amount z-score + card-testing probes, 7 features; `fraudkit/features.py:52-59`). Weekend + circular hour alone hurt ({ab.set_index('candidate').loc['+ weekend + circular hour', 'delta_vs_current_baseline']:.5f}). Source: `scripts/feature_ablation.py`, `tables/feature_ablation.csv`.

### 8.6 What improved between runs, and why
| Step | Change | Test PR-AUC | Other measured effects | Evidence for the "why" |
|---|---|---|---|---|
| Run 1 -> 2 | Replaced lifetime counter `card_txn_count` by `n_txn_30d` | {v1.test_pr_auc:.4f} -> {v2.test_pr_auc:.4f} | recall {pct(v1.recall)} -> {pct(v2.recall)}; precision {pct(v1.precision)} -> {pct(v2.precision)}; saved EUR {n(v1.saved_vs_no_model_eur)} -> {n(v2.saved_vs_no_model_eur)}; threshold {v1.threshold:.4f} -> {v2.threshold:.4f} | PSI {S['card_txn_count_psi_test_vs_fit']:.2f} for the retired feature (section 8.4); PROGRESS M8 |
| Run 2 -> 3 -> 4 | Same model retrained to store rule limits, decision policy data | unchanged ({v4.test_pr_auc:.4f}) | none | identical hash suffix `a6506be0`; PROGRESS M9, M10 |
| Run 4 -> 5/6 | 7 research-based features (28 inputs), Platt instead of isotonic calibration | {v4.test_pr_auc:.4f} -> {v6.test_pr_auc:.4f} | ROC-AUC {v4.test_roc_auc:.4f} -> {v6.test_roc_auc:.4f}; precision {pct(v4.precision)} -> {pct(v6.precision)}; recall {pct(v4.recall)} -> {pct(v6.recall)}; false alarms/10k {v4.false_alarms_per_10k_genuine:.1f} -> {v6.false_alarms_per_10k_genuine:.1f}; threshold {v4.threshold:.4f} -> {v6.threshold:.4f}; **total cost EUR {n(v4.total_cost_eur)} -> {n(v6.total_cost_eur)}** (higher by {n(v6.total_cost_eur - v4.total_cost_eur)}) | ablation table (section 8.5); calibration table (section 7.3) |

- Overall PR-AUC change run 1 -> final: {dpr:+.4f}. The 95% CIs of consecutive runs overlap (run 1 {v1.test_pr_auc_ci95_low:.4f}-{v1.test_pr_auc_ci95_high:.4f}; final {v6.test_pr_auc_ci95_low:.4f}-{v6.test_pr_auc_ci95_high:.4f}), so the improvement is small relative to the uncertainty and was measured on a single test set.
- **The money outcome did not improve in the last step:** total cost went up by EUR {n(v6.total_cost_eur - v4.total_cost_eur)} (saved vs no model {n(v4.saved_vs_no_model_eur)} -> {n(v6.saved_vs_no_model_eur)}), because the validation-chosen threshold moved (0.0238 -> 0.0837) and recall fell. The better ranking (PR-AUC, ROC-AUC) and the smooth probabilities are the gains; do not claim a cost reduction.
- Why the research features were added: scores in the app looked like only 0% or 100% (PROGRESS M11); literature-based features were then tested on validation. Source of the reasons: `fraudlens/PROGRESS.md` M8-M11 (documentation, cross-checked against the numbers above).

### 8.7 Limitations of the evaluation (be honest in the report)
1. **Simulated data** (Sparkov generator): patterns are cleaner than real fraud; the app's own Model report says real-world PR-AUC "would be much lower".
2. **One time split, one test period**; no repeated time-based validation. The 95% CI only reflects resampling of test rows.
3. **Cards overlap:** {card_ov} so the test does not measure unseen cardholders or cold start.
4. **Optuna CV is stratified and shuffled** (`train.py:453`), not time- or card-aware, so the CV score ({M['tuning']['best_cv_pr_auc']:.4f}) is not a time-honest estimate.
5. **Baselines are untuned** (logistic regression, random forest, default LightGBM) while LightGBM got 30 trials, so the leaderboard gap is partly tuning.
6. **Bootstrap CI treats rows as independent**, although a card's transactions are correlated.
7. **Money metric assumptions** (`fraudkit/costs.py`): flagging a fraud saves its whole amount, a false alarm costs only a flat EUR {M['threshold']['review_cost']:g} review, "no model" means zero protection; no customer-friction cost. "Saved vs no model" is therefore an upper-bound style figure.
8. **Calibration, threshold and rule limits all use the same validation period** (the test is the independent check).
9. **Labels are instantaneous** in the evaluation (no chargeback delay), except in the merchant/category-risk features.
10. **Policy rules** add {n(int(tab('policy_rules_impact_test.csv').iloc[1].flagged - tab('policy_rules_impact_test.csv').iloc[0].flagged))} reviews and no extra detected fraud on the test set (section 7.6).
11. **Weekly precision/recall in the Monitoring screenshot** falls to 0% in the last plotted week; this kit did not investigate (likely very few transactions in the partial last week: UNVERIFIED).
12. **Single machine, single seed (42).** Reproducibility is evidenced by identical metrics for the same hash (section 8.3), not by multiple seeds.

### 8.8 Other measurements (this machine; `_data/extra_facts.json`)
- Explaining one transaction with the served model, warm, mean of 20 calls: {X['latency_ms']['explain_one_transaction_warm_mean_of_20']:.1f} ms; probabilities for 10,000 prepared rows: {X['latency_ms']['predict_proba_10000_rows']:.1f} ms.
- Model file `fraudlens/artifacts/model.joblib`: {n(X['model_file']['bytes'])} bytes ({X['model_file']['n_estimators']} trees of up to {X['model_file']['num_leaves']} leaves).
"""


def s9() -> str:
    inv = pd.read_csv(REPO / "report_kit" / "_data" / "test_inventory.txt", sep=r"\s+", header=None, names=["tests", "file"])
    total = int(inv.tests.sum())
    ps = pytest_summary()
    files = py_files()
    groups = {"fraudkit/ (package)": [f for f in files if f.startswith("fraudlens/fraudkit/")], "app/ (Streamlit)": [f for f in files if f.startswith("fraudlens/app/")],
              "api/": [f for f in files if f.startswith("fraudlens/api/")], "tests/": [f for f in files if f.startswith("fraudlens/tests/")], "scripts/": [f for f in files if f.startswith("fraudlens/scripts/")]}
    rows = []
    for g, fs in groups.items():
        ph, co = loc(fs)
        rows.append({"folder": g, "python files": len(fs), "physical lines": ph, "code lines (non-blank, non-comment)": co})
    ph, co = loc(files)
    rows.append({"folder": "TOTAL (all tracked fraudlens/*.py)", "python files": len(files), "physical lines": ph, "code lines (non-blank, non-comment)": co})
    ruff = subprocess.run([str(FL / ".venv" / "Scripts" / "python.exe"), "-m", "ruff", "check", "."], cwd=FL, capture_output=True, text=True, check=False).stdout.strip().splitlines()[-1]  # noqa: S603
    fmt = subprocess.run([str(FL / ".venv" / "Scripts" / "python.exe"), "-m", "ruff", "format", "--check", "."], cwd=FL, capture_output=True, text=True, check=False).stdout.strip().splitlines()[-1]  # noqa: S603
    ci = [line.split("\t") for line in (REPO / "report_kit" / "_data" / "ci_runs.txt").read_text(encoding="utf-8").strip().splitlines()]
    return f"""## 9. Testing

### 9.1 Exact test count (command: `pytest --collect-only -q`, `_data/test_inventory.txt`)
**{total} tests collected.**

{md_table(inv, ['file', 'tests'], ['Test file (under fraudlens/)', 'Tests'])}

### 9.2 Result of the full run (command: `fraudlens/.venv/Scripts/python.exe -m pytest -q`, saved in `_data/pytest_full_run.txt`)
**{ps['passed']} passed, {ps['failed']} failed.** The failure is `tests/test_train_repro.py::test_training_is_reproducible`: `requests.exceptions.SSLError ... CERTIFICATE_VERIFY_FAILED` while contacting `api.kaggle.com` (the test calls `kagglehub`, which needs network access; a second run of that test alone failed the same way: `_data/pytest_repro_test_rerun.txt`). It is an environment (certificate/network) failure, not a model failure; an earlier run on the same day (console output only, no saved file) reported 81 passed. The test skips itself in CI because the dataset is not cached there.

### 9.3 Lint and format (commands run by `report_kit/scripts/facts_part2.py` in `fraudlens/`)
- `ruff check .` -> {ruff}
- `ruff format --check .` -> {fmt}
- Ruff rule sets selected in `fraudlens/pyproject.toml`: E, F, W, B, S (security), UP, SIM, PL; line length 120.

### 9.4 Continuous integration (`.github/workflows/fraudlens-ci.yml`)
Triggers: push or pull request touching `fraudlens/**`. Job on `ubuntu-latest`, Python 3.11: (1) checkout, (2) set up Python with pip cache, (3) `pip install -r requirements.txt -r requirements-api.txt ruff==0.17.1`, (4) `ruff check . && ruff format --check .`, (5) `pytest -q` (tests needing the Kaggle data or a trained model skip themselves).
Latest runs (command: `gh run list`, `_data/ci_runs.txt`): {'; '.join(f"{c[1]} on {c[2]} ({c[3][:10]})" for c in ci)}.

### 9.5 Lines of code (tracked Python files only; command: `git ls-files 'fraudlens/*.py'`, counted by `report_kit/scripts/facts_common.py:loc`)
{md_table(pd.DataFrame(rows))}

Not counted: Streamlit CSS inside `app/common.py` strings (counted as Python lines), YAML, Markdown, the Graphviz and report_kit files.
"""


def s10() -> str:
    g = tab("git_timeline.csv")
    phase = {
        "d490db0": "0. Original mini project (notebook, FraudShield app, docs)", "7ddcf5e": "1. FraudLens scaffold and data (M0-M1)", "df3077a": "1. FraudLens scaffold and data (M0-M1)",
        "b89f383": "2. Modelling and explanations (M3-M4)", "7e77aeb": "2. Modelling and explanations (M3-M4)", "e665984": "3. App, API and docs (M5-M7)", "788cafb": "3. App, API and docs (M5-M7)",
        "d1613e0": "3. App, API and docs (M5-M7)", "e406cda": "3. App, API and docs (M5-M7)", "3b4acd1": "4. Audit and enterprise hardening (M8)", "fbc2f86": "4. Audit and enterprise hardening (M8)",
        "c74513e": "4. Audit and enterprise hardening (M8)", "173feb0": "5. Policy rules and viva report (M9)", "9b7be1a": "6. Decision logic v2, analyst workflow, UI redesign (M10)",
        "ee5fcd8": "7. Model v3: features, calibration, risk score (M11)", "f73d82a": "8. Hosted-demo deployment preparation (M12)",
    }
    g["phase"] = g["hash"].map(phase)
    ph = g.groupby("phase").agg(first=("date", "min"), last=("date", "max"), commits=("hash", "count"), insertions=("insertions", "sum"), deletions=("deletions", "sum")).reset_index()
    return f"""## 10. Timeline (from `git log`, repository `gauravv-jainn/frauddet`, branch main)

{md_table(ph, None, ['Phase', 'First commit date', 'Last commit date', 'Commits', 'Lines added', 'Lines deleted'], {'insertions': N0, 'deletions': N0})}

All commits:

{md_table(g, ['hash', 'date', 'author', 'subject', 'files_changed', 'insertions', 'deletions'], ['Commit', 'Date', 'Author', 'Subject', 'Files', '+', '-'])}

{src('command: `git log --reverse --date=short --shortstat` (`tables/git_timeline.csv`)')}
Caveats: {len(g)} commits in total; phases 0-5 (13 commits) all carry the same calendar date (2026-10-20), so commit dates are not a reliable record of how long the work took. Authors in git: `Gaurav Jain` (1 commit, the original mini project) and `gauravv-jainn` (the rest); no commit is attributed to the second team member (see open questions). The 18,128 insertions of phase 7 include generated artifacts (`artifacts/runs/*`, tables, the viva PDF) and are not hand-written code. Today's date in this environment is 2026-11-01.
"""


def s11() -> str:
    sh = tab("screenshots_captions.csv")
    return f"""## 11. Screenshots: captions and which role sees what

All 12 files are copied to `figures/screenshots/` from `fraudlens/docs/screenshots/` (taken on 2026-10-27 right after the final model was saved, by `fraudlens/scripts/screenshots.py`). Every screenshot is signed in as an **admin** account named `demo.admin`, which the script creates (`scripts/screenshots.py:66`), so the sidebar shows every page. A viewer sees fewer pages and an analyst cannot open Admin; see `tables/pages_roles.csv`. The password never appears in an image. Captions marked "no" in the last column were written from the page's code, not from inspecting the image: check them against the image before printing.

{md_table(sh, ['file', 'page', 'roles_that_can_open_this_page', 'caption', 'image_inspected_for_caption'], ['File', 'Page', 'Roles that can open it', 'Caption (2-3 lines)', 'Image inspected?'])}
"""


def s12() -> str:
    return """## 12. Limitations, ethics, privacy, future scope

### 12.1 Limitations
- Evaluation limitations: section 8.7.
- Product limitations (README Limitations, Model report page, `SECURITY.md`): simulated data; no real merchants, devices, IPs or chargeback delays; no automated retraining or drift alerting; one flat review cost; local SQLite and in-process rate limiter; local passwords only; Streamlit and API run as separate processes sharing a file database.
- `model.joblib` is a pickle: it must only be loaded from storage that only the trusted training pipeline can write (`SECURITY.md`).
- The hosted demo is not live (section 6.7).

### 12.2 Ethics and fairness
- The model uses **gender** and **age** as inputs (`features_catalog.csv`: age is rank 9, gender rank 25 of 28 by mean |SHAP|) and the explanation text can quote them (example 3 in section 7.7: "Cardholder age 27 (higher-risk age group in the data)"). Using demographic attributes in a decision that affects customers can be legally and ethically problematic in real deployments. **No fairness or bias analysis was done** (searched the repository: nothing found). This must be stated as a limitation and future work.
- False alarms hurt customers: on the test set 406 genuine transactions were flagged (7.3 per 10,000 genuine) and the policy rules would add 408 more reviews (section 7.6). The cost model counts only the analyst's time, not customer friction.
- Explanations are model explanations (SHAP), not proof of fraud; the UI shows them with the calibrated probability and an action for a human analyst; analysts make the final decision on cases (Case queue).
- Responsible use: educational demo on synthetic data; not for real transactions (root README "Limits").

### 12.3 Privacy and security measures in the code
- Personal fields (first, last, street, zip, unix_time) are dropped on load (`fraudkit/data.py: clean`).
- Case store keeps only the last four digits of a card (`security.py:37-40`) and a SHA-256 hash for the blocklist (`security.py:30-34`). The hash is **unsalted and card numbers have a small search space, so it is pseudonymisation, not irreversible protection** (observation from reading the code).
- Passwords: PBKDF2 (600,000 iterations); API keys: only SHA-256 hashes stored; HTML escaping of data-derived text and CSV formula-injection protection (tests in `tests/test_platform.py`); JSON access logs without card numbers (PROGRESS M8).
- Audit trails: case audit and admin audit tables (section 6.4).
- Not in place: TLS, SSO/MFA, PCI DSS scope, secrets manager (`SECURITY.md`).

### 12.4 Future scope
From the repo's own lists (`SECURITY.md`, README Limitations): SSO/MFA and TLS, PostgreSQL, shared rate limiting, secrets manager, PCI scope review, model-governance approval before serving a retrained model, drift alerts wired to on-call, delayed-label handling, real merchant/device/IP features. Suggestions made for this kit (not in the repo): fairness audit of age/gender use; card-grouped and time-ordered cross-validation; tuning the baseline models; evaluating cold-start cards; a larger or real (with permission) dataset; multi-seed runs; deploying the prepared Hugging Face Space; re-running the base-paper notebook and saving its outputs.
"""


def s13() -> str:
    v = tab("viva_questions.csv")
    out = ["## 13. Viva: 25 questions with short answers\n", "Each answer ends with the artifact or code it comes from. Numbers are filled in by `report_kit/scripts/build_tables.py` from the artifacts. Machine-readable: `tables/viva_questions.csv`.\n"]
    for _, r in v.iterrows():
        out.append(f"**Q{r['no']}. {r['question']}**  \n{r['short_answer']}  \n*Reference:* {r['reference']}\n")
    return "\n".join(out)


def s14() -> str:
    return """## 14. OPEN QUESTIONS FOR THE STUDENTS (only you can answer these)

1. **Guide / mentor name** (template: "Faculty Incharge / Project Mentor"): NOT FOUND in the repository.
2. **HOD name** (if the college wants it on the certificate page): NOT FOUND.
3. **Academic year, semester, course name and course code, submission date, project group/batch number:** NOT FOUND. (The literature review in `docs/` shows class `D12B` and roll numbers; the batch/group number is not there.)
4. **Team members for the cover page.** `docs/Literature_Review_Fraud_Detection.docx` and the root `README.md` name Gaurav Jain (D12B, Roll 20) and Palash Matani (D12B, Roll 38), VESIT Mumbai. Confirm names, roll numbers, class/division and institute email IDs (the emails are in the literature review's title block; they are not copied here).
5. **Who did what (template section 5.5).** Git shows one author name for 15 of 16 commits (`gauravv-jainn`) and one commit by `Gaurav Jain`; nothing is attributed to the second member. Contribution evidence for the second member: NOT FOUND. Write the split yourselves and be able to show evidence (notebook parts, report sections, testing).
6. **College template choices:** which SDG(s) to claim and how to justify them (section 16 has verified wording for SDG 16 target 16.4 only); whether a logo is required; page/word limits; citation style (IEEE assumed); whether AI-assistance must be declared.
7. **Which version of the story to tell:** the original notebook mini-project (ULB data) vs FraudLens (Sparkov data). The base-paper comparison numbers come from the notebook, whose outputs are not saved: do you want to re-run it in Colab and keep the outputs?
8. **Hosted demo:** the Hugging Face deployment is prepared but not live. Do you want a live link in the report? If yes, deploy first and then re-check the URL.
9. **Final title** from the options in section 1.1, and whether to keep the name "FraudLens".
10. **Budget table in rupees (template 8.1):** hardware cost of your own laptop and electricity are NOT FOUND; the software and data cost is zero (open-source libraries, free Kaggle dataset, free-tier hosting is only prepared).
11. **Energy / sustainability measurement:** nothing was measured (no energy meter or profiler output in the repo). Decide whether to state "not measured".
12. **Screenshots:** the images are admin views with the account name `demo.admin`; decide whether to retake a viewer or analyst view for the report, and whether the images for Admin and Card 360 are acceptable (they show synthetic card endings and names only).
"""


def s15() -> str:
    inc = tab("inconsistencies.csv")
    return f"""## 15. INCONSISTENCIES found while checking numbers against artifacts and code

Rule used: a number was taken from `fraudlens/artifacts/` or from a command, never from README/PROGRESS. Where README/PROGRESS/code comments disagree, the artifact value is used and the disagreement is listed here (`tables/inconsistencies.csv`).

{md_table(inc, ['id', 'where', 'claim_or_value', 'evidence', 'severity'], ['ID', 'Where', 'Claim or value', 'Evidence', 'Severity'])}

Checks that agreed (no inconsistency): test PR-AUC, CI, ROC-AUC, precision, recall, flagged, cost and savings in the README Results block equal `metrics.json` (also recomputed independently from `test_scored.parquet` in section 17); the PROGRESS M3 validation leaderboard equals the first archived run; the README's "card_txn_count PSI 3.7-6.1 per test month" equals `tables/drift_card_txn_count_by_month.csv` (3.71-6.11).
"""


def s16() -> str:
    return f"""## 16. Template cross-reference and inputs for the sustainability / SDG / budget sections

The college template (`Common_Micro_Project_Report_Template_Computer_Engineering (1).docx`, VES Institute of Technology, Department of Computer Engineering) has sections 1-9, references and appendices. Where to find the facts:

| Template section | Facts in this file |
|---|---|
| Abstract | 1.2 |
| 1 Introduction | 2, 5 |
| 2 Problem definition, objectives, scope | 2 |
| 3 Literature review (table with Source / Approach / Strength-Limitation / Relevance) | 3, 4 (`tables/literature_survey.csv`) |
| 4 Proposed solution and system design (diagram, module table, data, algorithm) | 5, 6, 7; figures `architecture.png`, `scoring_flow.png`, `sqlite_er.png` |
| 5 Methodology and implementation (tools table, UI, team contribution) | 1.4, 7, 11, 14 (item 5) |
| 6 Testing, evaluation, results (test cases, metrics table) | 8, 9 |
| 7 Sustainability and SDG | below |
| 8 Feasibility, limitations, risks, ethics, privacy | 12 and the budget inputs below |
| 9 Conclusion and future scope | 8.6 and 12.4 |
| References | `references.md` |
| Appendices | `code_snippets.md`, `figures/screenshots/`, `tables/` |

### 16.1 Facts you can use for the sustainability assessment (nothing here is an energy measurement)
- **Compute:** training used CPU only (`n_jobs` = CPU count in `fraudkit/train.py:261`; no GPU in `metrics.json` machine info), {M['machine']['cpu_count']} threads; archived training run {n(RUNS.set_index('model_version').loc['20261025-233115-3abe0ebd', 'wall_clock_seconds'], 0)} s; the model file is {n(X['model_file']['bytes'] / 1e6, 2)} MB and one explained transaction takes about {X['latency_ms']['explain_one_transaction_warm_mean_of_20']:.0f} ms on this machine. **Energy use (kWh) and emissions: NOT MEASURED.**
- **Social:** 7.3 false alarms per 10,000 genuine transactions ({n(AT['fp'])} genuine test transactions flagged); explanations and an audit trail for every case; privacy measures in 12.3; fairness not assessed (12.2).
- **Economic (model-side, simulated data):** missed fraud EUR {n(AT['missed_amount'], 0)} and review cost EUR {n(AT['review_cost_total'], 0)} against EUR {n(AT['fraud_amount'], 0)} of fraud in the test period (section 8.1); assumptions in 8.7 item 7.
- **SDG candidates:** the UN page https://sdgs.un.org/goals/goal16 states target 16.4: "By 2030, significantly reduce illicit financial and arms flows, strengthen the recovery and return of stolen assets and combat all forms of organized crime" (text fetched and matched). Whether card-fraud detection supports 16.4 directly is for you to argue; the project does not measure that. Wording of targets for any other SDG (for example 8 or 9) was NOT verified here.
- **Trade-offs already visible:** more review (higher recall) means more customer friction and analyst cost (threshold and expected-loss tables in section 7.5).

### 16.2 Inputs for the budget table (template 8.1; amounts in rupees are NOT FOUND)
| Item | Fact | Source |
|---|---|---|
| Dataset | free public Kaggle dataset | `fraudkit/data.py` |
| Software | open-source Python libraries (section 1.4) | `requirements*.txt` |
| Hosting | none running; a free Hugging Face Space is prepared (PROGRESS M12 says the free tier offers 16 GB RAM; not re-verified in this kit) | `deploy/space`, `PROGRESS.md` M12 |
| Hardware / electricity | the students' own computer; cost NOT FOUND | - |
"""
