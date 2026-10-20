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
