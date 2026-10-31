# FraudLens build progress

Machine: 32 logical CPUs, Windows 11, Python 3.11.15 (installed with uv; system Python is 3.12).

## M0 - Repo and setup (done)
- Cloned `gauravv-jainn/frauddet` (main). Existing files (notebook, docs/, fraudshield/, README.md) untouched; everything new lives in `fraudlens/`.
- `.venv` created with uv, pinned `requirements.txt` (direct deps) and `requirements-lock.txt` (full freeze).
- `configs/default.yaml` holds seed, validation fraction, Optuna trials, review cost, risk bands.

## M1 - Data (done)
- Source used: **Kaggle `kartik2112/fraud-detection`** via kagglehub (public download worked without credentials). No fallback needed.
- Provided time split kept: train 2019-01-01 -> 2020-06-21 12:13, test 2020-06-21 12:14 -> 2020-12-31.

| split | rows | cards | frauds | fraud rate |
|---|---|---|---|---|
| train | 1,296,675 | 983 | 7,506 | 0.5789% |
| test | 555,719 | 924 | 2,145 | 0.3860% |

- Dropped columns not used by the model: first, last, street, zip, unix_time. The `fraud_` prefix Sparkov puts on every merchant name is stripped.

## M2 - Leakage-free features (done)
- `fraudkit/features.py`: 19 numeric + 2 categorical features. Card-history features (velocity 1h/24h, 24h spend, amount vs expanding card mean and card-category mean, hours since previous, geo-speed + impossible-travel flag, first time in category, history length) use only strictly earlier rows of the same card.
- Merchant: smoothed target encoding, out-of-fold on the fit split, mapping fitted on the fit split only.
- Tests (`tests/test_features.py`, 6 pass): truncation test (features of row i unchanged when all later rows removed), future-perturbation test, history-scoring equals batch, first-row semantics, haversine sanity.
- Full build on 1.85M rows: 2.7 s. Observed signal on train: fraud median amount = 5.23x card mean (genuine 0.66x); night fraud rate 2.09% vs 0.12% day. Distance home->merchant has no signal in this simulator (medians 78 km for both).

## M3/M3b - Modelling and visible, reproducible training (done)
- `python -m fraudkit.train --config configs/default.yaml` (flags `--trials`, `--fast`, `--out`, `--no-live`). Rich live view: stage bars + live Optuna table; leaderboard printed at the end. Uses all 32 CPU threads.
- Logged to `artifacts/training_log.jsonl`; Optuna history in `artifacts/trials.csv`; all results in `artifacts/metrics.json`.
- `--fast` smoke run: ~25 s. Full run (30 trials) ran in the visible terminal: **612 s**.
- Split: fit 1,102,173 rows / validation 194,502 rows (1,133 frauds, from 2020-04-03 17:54) / test 555,719.

| model (validation) | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| LightGBM tuned (Optuna) (selected) | 0.9845 | 0.9997 | 0.00049 |
| Random forest | 0.9540 | 0.9970 | 0.00110 |
| Logistic regression | 0.7411 | 0.9682 | 0.00264 |
| LightGBM default | 0.7247 | 0.9480 | 0.00195 |

- Optuna best CV PR-AUC 0.9841. LightGBM with default params is unstable on this imbalance (PR-AUC below logistic regression); reported as-is.
- Calibration: isotonic (1133 validation positives). Brier validation 0.00049 -> 0.00039; test 0.00043 -> 0.00040.
- Money threshold (review cost 5): **0.1111** chosen on validation.
- **Test (once):** PR-AUC 0.9682 (95% bootstrap CI 0.9620-0.9743, 1000 resamples), ROC-AUC 0.9978, recall 0.956, precision 0.855, false alarms 6.3 per 10k genuine, flagged 2,397. Fraud amount 1,133,325; cost with model 31,104; **saved 1,102,221** vs no model.
- Reproducibility test (`tests/test_train_repro.py`) trains twice with `--fast` and compares metrics. Only difference found was the last float bit (1e-16) of random-forest ROC-AUC from parallel averaging, so the test compares to 10 decimals.

## M4 - Explanations (done)
- `fraudkit/explain.py`: SHAP TreeExplainer (log-odds) per transaction. Features are grouped (e.g. amt+log_amt, 1h/24h velocity+24h spend, geo-speed+impossible-travel) so each fact appears once; groups below 0.05 log-odds are not reported.
- Up to 4 raising and 2 lowering reasons, worded from the actual values (amount vs card average, amount vs card's category average, night time, velocity, minutes since previous, km and km/h for travel, first use of a category, merchant fraud history).
- Wording follows the value, not just the SHAP sign (a small amount that raises risk says "very small amount ... test charges", not "large amount").
- Risk band from calibrated probability: Medium starts at the money threshold (config `bands.medium: null`), High 0.30, Critical 0.70; one-sentence summary and action (approve / step-up / hold for review / block and call).
- What-if: re-score with edited amount, hour or merchant distance (distance moves the merchant along the same bearing); velocity features are recomputed from the card's history.
- Tests (`tests/test_explain.py`): SHAP + base value reconstructs the LightGBM raw margin (atol 1e-6); every reason's features contribute in the stated direction above the cut-off; what-if with no edits equals batch scoring.

## M5 - Streamlit app (done)
- `streamlit run app/streamlit_app.py` (st.navigation over `app/views/`; the folder is not called `pages/` because Streamlit's auto-discovery bypassed the navigation on direct URLs).
- Six pages: Dashboard, Check a transaction (gauge, band, reason cards, what-if), Batch scoring (schema validation, risk-ranked table, CSV download), Case queue (SQLite, Confirm/Genuine/Escalate, audit trail, agreement stats), Model report, Training Lab (timeline, trial history, Retrain subprocess with live log tail; button disabled while a run is active).
- Sidebar threshold (validated value by default) and review cost update every page. Theme in `.streamlit/config.toml`; server bound to localhost.
- Fixes made after looking at the pages: KPI values were truncated at 1280 px (compact EUR format, shorter labels), the reliability chart used equal-count bins that all sat near 0 (now fixed probability bins), the cost curve needed a log y-axis, and the Retrain button briefly showed the previous run as finished (status is now written at launch).
- Batch scoring of a CSV round-trip of 500 test rows reproduces the stored test scores exactly (max abs diff 0.0).

## M6 - Agency touches (done)
- P1: all thresholds/costs/bands in `configs/default.yaml`; Demo mode streams test transactions in time order into the case queue.
- P2: FastAPI `POST /score` (`api/main.py`) with `X-API-Key` header (env `FRAUDLENS_API_KEY`, constant-time compare), returns probability, band, flag, action, summary and reasons; `Dockerfile` + `requirements-api.txt`. API scores equal the offline scores (tested). Docker image not built here (Docker is not part of this setup).

## M7 - Verification (done)
- `pytest`: 23 passed (features/leakage 6, explanations 5, cases 1, reproducibility 1, app click-through 7, API 3).
- AppTest drives all 6 pages headless (what-if submit, manual form, batch sample, queue add/confirm/demo stream) with no exceptions, and checks dashboard KPIs against `artifacts/metrics.json`; a direct test asserts the app's metric dict equals `metrics.json["test"]["at_threshold"]` exactly.
- Final full training run repeated in the terminal after the reliability-table change: every metric identical to the first full run (full-scale reproducibility).
- Playwright screenshots of all pages in `docs/screenshots/` (`scripts/screenshots.py`), reviewed visually.
- README results table generated from `metrics.json` by `scripts/results_table.py`.

## M8 - Audit and enterprise hardening (done)
**Audit findings (all fixed, each with a regression test):**
- Stored XSS: merchant/city/reason text from an uploaded CSV was rendered as raw HTML in the case queue and Check page. All data-derived text is now escaped.
- CSV formula injection in every CSV download: cells starting with `= + - @` are now prefixed with `'`.
- Retrain overwrote `model.joblib`/parquet in place, so readers could load a half-written file. All artifacts are now staged and swapped atomically.
- The batch-scoring cache was not keyed on the model, so it served stale scores after a retrain.
- API and app could disagree on risk bands (model file vs config). Both now use one serving loader that reads the config.
- SQLite without WAL/busy-timeout ("database is locked" with concurrent users).
- `inf` ratio features possible from uploads; a malformed `is_fraud` column crashed batch scoring; the manual form listed only 400 of 924 cards; `" | "`-joined reasons broke on merchant names containing a pipe.
- Sidebar threshold/review cost silently reset on page switches (Streamlit drops widget state). Dashboard numbers could have been computed with cost 0.
- Unassigned cases showed "assignee nan", and "Unassign" was offered for them.
- **Model defect found by the new drift monitor:** `card_txn_count` is a lifetime counter that only grows (PSI 3.7-6.1 every test month, 5th most important feature). Replaced with stationary `n_txn_30d`. Test PR-AUC 0.9682 -> **0.9722** (CI 0.9666-0.9773), recall 95.6% -> 96.4%, saved 1,102,221 -> 1,104,591, precision 85.5% -> 81.1%, threshold 0.1111 -> 0.0238.

**Platform added:**
- Login with PBKDF2 (600k iterations), lockout, session timeout, forced change of admin-set passwords, first-run admin setup; roles viewer/analyst/admin enforced server-side on every page and action.
- Admin page (users, API keys shown once and stored hashed, audit log with export, system status, model versions) and My account page.
- Case queue: assignment, "my cases", case age, CSV export, actor recorded on every decision, card numbers stored as last 4 digits only, admin-only clear with confirmation (logged).
- Monitoring page: monthly PSI drift per feature and for the score vs the training reference profile, alerts, weekly precision/recall/flag rate.
- Versioned models (`<UTC time>-<sha256>`), run archive in `artifacts/runs/`, model version on API responses and on cases.
- API `/v1` (score, model, health, ready), DB-backed API keys plus bootstrap env key, per-key rate limit (429), batch limit (413), request IDs, JSON access logs without card numbers, hot model reload, legacy routes kept.
- Validated config (pydantic, fails fast), DB migrations (an old case DB is upgraded in place, tested), operator CLI `python -m fraudkit.admin`.
- Ops: non-root Dockerfile with healthcheck, docker-compose (localhost-only ports), `.env.example`, `pyproject.toml` (ruff + pytest), GitHub Actions CI (lint + tests; data-dependent tests skip), SECURITY.md.
- Tests: 61 passing (was 23). `ruff check` and `ruff format --check` are clean.
- Final model: version `20261020-154750-a6506be0`, full run 502 s.

## M9 - Policy rules and viva report (done)
- Found in the app: a €12,000 purchase 2,500 km from home scored 0.0% (Low). Cause: in the training data every fraud is <= €1,376, amounts above €2,000 are always genuine (731 rows, 0 frauds), and the maximum home-merchant distance is 152 km. Trees cannot extrapolate.
- Added `fraudkit/rules.py`: configurable policy rules that can only raise the band (large amount >= 2,000; spending spike >= 20x the card average and >= 1,500; >= 500 km from home; impossible travel with a >= 500 km jump; >= 5 transactions in 1 h; key feature outside the training range). Wired into the app (purple rule cards, "model alone -> raised to"), batch scoring, the case-queue demo stream and the API (`model_band`, `policy_rules`).
- Measured on the test set: model alone 2,548 flags / recall 96.36% / cost 28,734; with rules 3,159 flags / recall 96.36% / cost 31,789 (+611 reviews, no extra fraud in simulated data). Two looser rule versions were measured and rejected (impossible-travel flag: 15,074 hits for 1 extra fraud; 10x spending spike: 2,398 extra reviews for 10 frauds).
- Retrained to store training ranges in the reference profile: identical model (same sha256 prefix a6506be0), version 20261020-172652-a6506be0.
- Amount input limit raised (it rejected 250,000).
- `docs/FraudLens_Viva_Report.pdf` (10 pages) generated by `scripts/build_report.py` from the artifacts: algorithms and where they are used, comparison protocol and results, tuning, calibration, threshold, test results with CI, SHAP, drift finding, policy rules, limitations, 26 viva Q&As, glossary.
- Tests: 66 passing; ruff clean.

## M10 - Decision logic v2, analyst workflow, UI redesign (done)
- **Card-profile features** (amount vs card's previous max, category share, hour share): two-seed validation ablation, no gain beyond seed noise (0.0005), so not added to the model; kept as context (`scripts/feature_ablation.py`, `artifacts/feature_ablation.json`).
- **Expected-loss decision policy** (`fraudkit/decision.py`): review when p x amount >= k x review cost, k tuned on validation (k = 3). Validation cost: threshold 8,838 vs expected loss 9,338, so the threshold is served. On test, expected loss would cost 24,473 vs 28,734 at 92.8% vs 96.4% recall; not switched, because that would be tuning on test. Actions: approve / step-up / review / block.
- **Rule limits from data:** review amount = 1.25 x max fit-period fraud amount (1,371.81 -> 2,000); far from home = 2 x max fit distance (-> 400 km); spending spike tuned on validation (50x, >= 2,000). Medium rules now mean step-up, High/Critical mean review/block.
- **Watchlists:** blocked cards (SHA-256 hash + masked label) -> Critical/block; watched merchants -> High/review; confirming fraud blocks the card (config `rules.block_confirmed_cards`). DB migration 3 adds `watchlist`, `case_comments` and decision columns on cases.
- **Analyst workflow:** board/table queue, bulk actions, SLA/overdue, comments, priority by expected loss, Card 360 page (timeline, map, spending profile, cases), sidebar inbox.
- **UI:** dark sidebar with logo, sectioned navigation with icons, page header with model/policy badges, dashboard filters (period, category, state) with daily trend, decision mix, alert heatmap, top merchants.
- Fixed while reviewing: card unblock by masked label could hit a different card with the same last 4 digits (now by hash); case detail defaulted to a closed case; clipped page titles; truncated labels; a negative chart axis.
- Retrain reproduced the identical model (sha a6506be0), version 20261025-224238-a6506be0. Tests: 76 passing; ruff clean. Viva report updated (section 10b).

## M11 - Model v3: research-based features, smooth calibration, risk score (done)
- **Problem reported:** scores were only 0% or 100%. Measured: 95.07% of test transactions got exactly 0, 1,539 got exactly 1, 94 distinct calibrated values. Cause: isotonic calibration (step function); the raw model had 555,719 distinct scores.
- **Research** (sources in the viva report, section 15): RFM spending windows and terminal risk with label delay (Fraud Detection Handbook), velocity counters (US patent 8041620, RisingWave), card testing (Chargeback Gurus, Fraudlogix), amount/elapsed time and log z-score (arXiv 1909.01185), circular time (arXiv 2503.20477). Device/IP/BIN signals are not in this dataset.
- **Features** (leakage-tested; merchant risk ignores labels younger than 7 days, with its own test). Validation ablation, two seeds: all research groups 0.9864 vs 0.9847 (noise 0.0005); the simplest equal set was kept (spending windows + amount z-score + card-testing probes, 7 features). Weekend/circular hour alone hurt (0.9839).
- **Calibration:** isotonic / Platt / beta compared on two time-halves of validation by log-loss; Platt chosen (isotonic worst). Floor lowered from 1e-6 to 1e-12 after a test showed the floor itself merged tiny scores. Test: 522,809 distinct probabilities.
- **Risk score 0-999** on a log-odds scale (p = 1e-6 -> 0, p = 0.999 -> 999, ~33 points doubles the odds). A first rank-based version saturated at the decision threshold (threshold = score 998) and was replaced. Now: genuine median 25, fraud median 950, review threshold 551, High 625, Critical 707.
- **Results** (same untouched test set): PR-AUC 0.9722 -> 0.9765 (CI 0.9719-0.9811), ROC-AUC 0.9978 -> 0.9995, precision 81.1% -> 83.5%, false alarms/10k 8.7 -> 7.3, recall 96.4% -> 95.9%, total cost 28,734 -> 30,137 (threshold chosen on validation moved 0.0238 -> 0.0837).
- Final model version 20261026-184402-d37e9778 (metrics identical to run 20261025-233115-3abe0ebd; the final run's wall-clock time includes the machine sleeping). Tests: 79 passing.

## M12 - Hosted demo deployment (done; going live needs the owner's Hugging Face account)
- Target: free Hugging Face Docker Space. It offers 16 GB of RAM; the container used about 2.4 GB in a local run with the real artifacts, more than the 1 GB limit commonly quoted for Streamlit Community Cloud.
- `scripts/deploy_space.py` uploads code to a public Space and the model and data to a private dataset repo (nothing added to GitHub).
- `fraudkit/bootstrap.py` downloads the artifacts and creates the accounts at start-up, and refuses to start without passwords so a public URL can never show the first-run admin setup.
- Verified locally in the built container: refuses to start without passwords, demo sign-in, dashboard, 500-row batch scoring. The download path and the live Space itself are not exercised until the first real deploy.
