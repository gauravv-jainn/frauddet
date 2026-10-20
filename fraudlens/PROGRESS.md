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
