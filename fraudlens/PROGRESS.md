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
