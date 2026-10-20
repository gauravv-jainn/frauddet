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
