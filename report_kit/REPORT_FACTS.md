# REPORT_FACTS: FraudLens report kit

Generated on 2026-11-01 from the repository state at commit `f73d82a` (branch main). Not a report: facts, numbers and sources for the college report book. **Rules followed:** every number comes from `fraudlens/artifacts/` or from a command named next to it (numbers quoted from documents, such as the base-paper table, say so); anything not found says NOT FOUND; anything not verified says UNVERIFIED; disagreements are in section 15. Nothing was retrained and no file outside `report_kit/` was changed.

Contents: 1 Title, abstract, tech stack | 2 Problem and scope | 3 Base paper | 4 Literature survey | 5 Dataset | 6 Architecture | 7 Methodology | 8 Results | 9 Testing | 10 Timeline | 11 Screenshots | 12 Limitations, ethics, privacy, future | 13 Viva questions | 14 Open questions | 15 Inconsistencies | 16 Template cross-reference | 17 Self-check log

## 1. Title options, abstract, keywords, tech stack

### 1.1 Title options (suggestions written for this kit, not facts)
1. FraudLens: An Explainable, Calibrated Credit Card Fraud Detection System with an Analyst Workflow
2. Explainable Credit Card Fraud Scoring with Calibrated Risk Scores and Cost-Aware Decisions
3. From Paper to Product: Re-evaluating and Extending Machine-Learning Credit Card Fraud Detection
4. Credit Card Fraud Detection Using LightGBM, SHAP and a Case-Management Application

### 1.2 Abstract draft (190 words; every number is from `fraudlens/artifacts/metrics.json`)
Credit card fraud is rare (0.39% of the 555,719 test transactions here) and costly, so accuracy is a misleading measure and every alert must be justified to a human analyst. This project reproduces and critiques the evaluation of Tanouz et al. (2021) and then builds FraudLens, an explainable fraud-scoring system. A leakage-free per-card feature pipeline (28 features) feeds a LightGBM model tuned with Optuna on PR-AUC and trained on a time-ordered split of the Kaggle Sparkov simulated transactions. Platt-scaled probabilities, chosen by out-of-sample log-loss, replace an isotonic calibrator that produced scores of only 0% or 100%, and a 0-999 log-odds risk score makes them readable. A money-aware threshold, configurable policy rules for out-of-range cases and SHAP-based plain-English reasons turn scores into approve, step-up, review or block decisions. The system is delivered as a Streamlit analyst application with role-based access, a SQLite case queue with audit trail, PSI drift monitoring and a FastAPI scoring service. On the untouched test period the model reaches PR-AUC 0.9765 (95% CI 0.9719-0.9811), 95.9% recall and 83.5% precision at 7.3 false alarms per 10,000 genuine transactions. All results come from simulated data and a single split.

### 1.3 Keywords
credit card fraud; class imbalance; LightGBM; probability calibration; SHAP explanations; precision-recall; population stability index (concept drift); decision support.

### 1.4 Tech stack with exact versions
| Category | Component | Version | Source |
|---|---|---|---|
| Language | Python | 3.11.15 | metrics.json versions |
| Data | pandas | 3.0.6 | metrics.json versions |
| Data | numpy | 2.4.6 | metrics.json versions |
| ML | scikit-learn | 1.9.1 | metrics.json versions |
| ML | lightgbm | 4.7.0 | metrics.json versions |
| ML | shap | 0.51.0 | metrics.json versions |
| ML | optuna | 5.0.0 | metrics.json versions |
| UI | streamlit | 1.66.0 | metrics.json versions |
| UI | plotly | 7.1.0 | metrics.json versions |
| Core | joblib | 1.6.0 | fraudlens/requirements.txt |
| Core | kagglehub | 1.0.2 | fraudlens/requirements.txt |
| Core | pydantic | 2.14.1 | fraudlens/requirements.txt |
| Core | pyyaml | 6.0.3 | fraudlens/requirements.txt |
| Core | rich | 15.0.0 | fraudlens/requirements.txt |
| Test | pytest | 9.1.1 | fraudlens/requirements.txt |
| API | fastapi | 0.153.0 | fraudlens/requirements-api.txt |
| API | uvicorn | 0.54.0 | fraudlens/requirements-api.txt |
| API | httpx | 0.28.1 | fraudlens/requirements-api.txt |
| Quality | ruff | 0.17.1 | fraudlens/requirements-dev.txt and .github/workflows/fraudlens-ci.yml |
| Tooling | playwright (screenshots) | 1.64.0 | fraudlens/requirements-dev.txt |
| Tooling | matplotlib | 3.11.2 | fraudlens/requirements-dev.txt |
| Tooling | pypdfium2 | 5.14.0 | fraudlens/requirements-dev.txt |
| Deploy | huggingface_hub (deploy script) | 2.4.0 | fraudlens/requirements-deploy.txt |
| Deploy | Docker base image | python:3.11-slim | fraudlens/Dockerfile |
| Deploy | Docker engine | 29.6.1 | command: docker info --format '{{.ServerVersion}}' (run in this session) |
| Storage | SQLite (runtime library) | 3.50.4 | command: python -c 'import sqlite3; print(sqlite3.sqlite_version)' |
| Storage | pyarrow (parquet) | 26.0.0 | command: importlib.metadata.version('pyarrow') in fraudlens/.venv |
| Diagrams | Graphviz dot | 15.0.0 | command: dot -V |
| Hardware | CPU threads used for training | 32 | metrics.json machine |
| Hardware | Processor | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel | metrics.json machine |
| OS | Training machine platform | Windows-10-10.0.26300-SP0 | metrics.json machine |

*Source:* `fraudlens/artifacts/metrics.json` -> versions, machine; `fraudlens/requirements*.txt`; commands noted in the table


## 2. Problem, motivation, objectives, scope, out-of-scope

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

## 3. Base paper (Tanouz et al.): summary, weaknesses, numbers vs ours

**Citation (metadata VERIFIED in Crossref, DOI 10.1109/iciccs51141.2021.9432308):** D. Tanouz, R. R. Subramanian, D. Eswar, G. V. P. Reddy, A. R. Kumar, and C. V. N. M. Praneeth, "Credit Card Fraud Detection Using Machine Learning," 2021 5th International Conference on Intelligent Computing and Control Systems (ICICCS), 2021, pp. 967-972.

**What the paper says (abstract, via Semantic Scholar):** card fraud is growing; the work proposes machine-learning classifiers such as logistic regression, random forest and Naive Bayes for a heavily imbalanced dataset and computes accuracy, precision, recall, F1, confusion matrix and ROC-AUC.

**Pipeline as summarised by the team's review** (`docs/Literature_Review_Fraud_Detection.docx` section 2; the full paper is behind an IEEE paywall, so this is UNVERIFIED against the paper): compare Decision Tree, Random Forest, Logistic Regression and Naive Bayes on the ULB dataset; normalise Time and Amount; under-sample the genuine class to 1:1; drop features correlated with the label; remove outliers; only then split train/test. Reported best model: Random Forest with 96.77% accuracy, 100% precision, 91.11% recall (UNVERIFIED against the paper).

**Weaknesses found by the team** (same document; the team's analysis, not independently re-run in this kit):
1. The test set is balanced and tiny: the reported metrics are consistent with about 124 test rows, about 45 of them fraud, while the real fraud rate is 0.17%.
2. Preprocessing (under-sampling, label-based feature selection, outlier removal) happens before the split, which is data leakage.
3. The reported "ROC-AUC" matches balanced accuracy within about 0.01, as if computed from hard 0/1 predictions.
4. Differences between models are a few transactions, with no confidence intervals.

### Paper numbers vs reproduced numbers vs FraudLens
| Setting | Dataset | Test rows | Precision % | Recall % | False alarms | Source |
|---|---|---|---|---|---|---|
| Paper as reported [Tanouz 2021] | ULB (Kaggle creditcard) | ~124, balanced | 100.0 | 91.11 | 0 | docs/Literature_Review_Fraud_Detection.docx Table 1; UNVERIFIED against the paper (paywalled) |
| Team's re-implementation, paper-style (Random Forest) | ULB | 247, balanced | 94.68 | 86.41 | 5 | same table; notebook has no saved outputs so NOT re-run in this kit |
| Random Forest, under-sampled training, real test | ULB | 56,962 (98 fraud) | 4.3 | 91.84 | 2004 | same table; NOT re-run in this kit |
| Random Forest + SMOTE, real test | ULB | 56,962 (98 fraud) | 88.3 | 84.69 | 11 | same table; NOT re-run in this kit |
| FraudLens tuned LightGBM at the validated threshold | Sparkov simulated (Kaggle kartik2112) | 555,719 (2,145 fraud) | 83.52 | 95.9 | 406 | fraudlens/artifacts/metrics.json -> test.at_threshold (different dataset: NOT directly comparable to the rows above) |

**Read this before using the table.**
- Rows 1-4 are copied from `docs/Literature_Review_Fraud_Detection.docx` Table 1. `CreditCard_Fraud_MiniProject.ipynb` has **no saved cell outputs** (0 of its code cells have outputs), so these numbers could not be regenerated from an artifact here: **NOT re-verified**. Re-run the notebook in Colab (and keep the outputs) before printing them in the report.
- Row 5 uses a different dataset (Sparkov simulated, 555,719 test rows) and a different protocol, so it must not be presented as "better than the paper".
- The paper's own numbers could not be checked (paywall): UNVERIFIED.

## 4. Literature survey (every reference verified against a registry)

Verification method (`report_kit/scripts/verify_refs.py`, `verify_refs2.py`): each candidate DOI was looked up in Crossref and kept **only if** the registry record matched the title (similarity >= 0.90), first author and year; arXiv records through the arXiv API; web pages by fetching them. **No DOI in this kit was written from memory without registry confirmation.** Where an abstract was not available, the "method / dataset / result / limitation" cells say NOT STATED instead of guessing. Full bibliographic entries: `references.md`; machine-readable: `tables/literature_survey.csv`, `tables/references_verified_final.csv`.

| # | Authors | Year | Venue | Method | Dataset | Result | Limitation | Verification |
|---|---|---|---|---|---|---|---|---|
| 1 | Tanouz et al. | 2021 | 2021 5th International Conference on Intelligent Computing and Control Systems (ICICCS) | Classifiers named in the abstract: logistic regression, random forest, Naive Bayes (the team's review also lists a decision tree: see INCONSISTENCIES) | ULB card dataset (per team's own review: docs/Literature_Review_Fraud_Detection.docx) | Random Forest best with 96.77% accuracy, 100% precision, 91.11% recall (as quoted in the team's review; UNVERIFIED against the paper, which is paywalled) | Per the team's review: tiny balanced test set (~124 rows), preprocessing before the split, 'ROC-AUC' looks like balanced accuracy | metadata VERIFIED (Crossref); numbers UNVERIFIED |
| 2 | Bhattacharyya et al. | 2011 | Decision Support Systems | SVM, random forest and logistic regression compared (per team's own review: docs/Literature_Review_Fraud_Detection.docx) | Real credit-card transactions (per team's review) | Random forest strongest overall, with a precision-recall trade-off (per team's review) | NOT STATED (abstract not retrievable; full text not read) | metadata VERIFIED (Crossref); content from team's review |
| 3 | Randhawa et al. | 2018 | IEEE Access | Standard classifiers, then hybrid AdaBoost and majority voting | A public credit-card dataset, then a real-world dataset from a financial institution, with noise added | Majority voting gives good accuracy in detecting fraud (no numbers in the abstract) | NOT STATED in the abstract; full text not read | VERIFIED (Crossref) |
| 4 | Dal Pozzolo et al. | 2014 | Expert Systems with Applications | Practitioner view: imbalance, concept drift and limited investigator capacity; recommends ranking-based metrics such as precision among top alerts (per team's review) | NOT STATED | NOT STATED | NOT STATED | metadata VERIFIED (Crossref); content from team's review |
| 5 | Dal Pozzolo et al. | 2015 | 2015 IEEE Symposium Series on Computational Intelligence | Under-sampling shifts predicted probabilities, so they must be re-calibrated (per team's review) | NOT STATED | NOT STATED | NOT STATED | metadata VERIFIED (Crossref); content from team's review |
| 6 | Dal Pozzolo et al. | 2018 | IEEE Transactions on Neural Networks and Learning Systems | Realistic modelling of a fraud-detection system (delayed supervised information, alert-based performance measures) and a new learning strategy | Data from an industrial partner (abstract) | NOT STATED in the part of the abstract read | NOT STATED in the abstract; full text not read | VERIFIED (Crossref); not cited in the repo's docs |
| 7 | He and Garcia | 2009 | IEEE Transactions on Knowledge and Data Engineering | Survey of learning from imbalanced data: sampling, cost-sensitive and algorithm-level remedies (per team's review) | N/A (survey) | N/A (survey) | NOT STATED | metadata VERIFIED (Crossref); content from team's review |
| 8 | Chawla et al. | 2002 | Journal of Artificial Intelligence Research | SMOTE: synthetic minority over-sampling combined with under-sampling the majority class | NOT STATED in the abstract | Better classifier performance in ROC space than under-sampling alone (abstract) | NOT STATED in the abstract | VERIFIED (Crossref) |
| 9 | Fernandez et al. | 2018 | Journal of Artificial Intelligence Research | 15-year review of SMOTE and its extensions | N/A (review) | SMOTE is a de facto standard for imbalanced learning (abstract) | Can add noise and gives little gain for strong tree ensembles on some datasets (per team's review) | VERIFIED (Crossref) |
| 10 | Saito and Rehmsmeier | 2015 | PLOS ONE | Compares ROC and precision-recall plots on imbalanced data | Imbalanced datasets (bioinformatics context in the abstract) | ROC plots can be deceptive on imbalanced data; PR plots are more informative (abstract) | NOT STATED in the abstract; full text not read | VERIFIED (Crossref) |
| 11 | Davis and Goadrich | 2006 | Proceedings of the 23rd international conference on Machine learning - ICML '06 | Relationship between precision-recall and ROC curves: a curve dominates in ROC space iff it dominates in PR space (per team's review) | N/A (theory) | NOT STATED | NOT STATED | metadata VERIFIED (Crossref); content from team's review |
| 12 | Kaufman et al. | 2012 | ACM Transactions on Knowledge Discovery from Data | Formal definition of data leakage and how to detect and avoid it (per team's review) | N/A | N/A | NOT STATED | metadata VERIFIED (Crossref); content from team's review |
| 13 | Ribeiro et al. | 2016 | Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining | LIME: explains one prediction of any classifier with a local interpretable model | Text and image classifiers (not card fraud) | NOT STATED in the part of the abstract read | Explanations of PCA-anonymised features are not business-interpretable (team's note) | VERIFIED (Crossref) |
| 14 | Breiman | 2001 | Machine Learning | Random forests: combination of tree predictors; generalisation error converges as trees grow | N/A | N/A | NOT STATED in the abstract; full text not read | VERIFIED (Crossref) |
| 15 | Chen and Guestrin | 2016 | Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining | XGBoost: scalable tree boosting with a sparsity-aware algorithm | NOT STATED in the part of the abstract read | NOT STATED in the part of the abstract read | NOT STATED in the abstract; full text not read | VERIFIED (Crossref); used in the original notebook, not in FraudLens |
| 16 | Akiba et al. | 2019 | Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining | Optuna: define-by-run hyper-parameter optimisation framework with pruning | N/A | N/A | NOT STATED in the abstract; full text not read | VERIFIED (Crossref); the tool used for tuning in fraudkit/train.py |
| 17 | Niculescu-Mizil and Caruana | 2005 | Proceedings of the 22nd international conference on Machine learning - ICML '05 | Probability calibration of supervised learners (title; abstract not retrieved) | NOT STATED | NOT STATED | NOT STATED | VERIFIED (Crossref); content NOT read |
| 18 | Lucas et al. | 2019 | arXiv | Hidden Markov models over sequences of amounts and elapsed time give extra features for card-fraud detection | NOT STATED in the part of the abstract read | NOT STATED | NOT STATED | VERIFIED (arXiv record); preprint |
| 19 | Stotsky | 2025 | arXiv | Moving-window confidence-interval method and a fraud-risk scoring method for detecting fraud attacks | A card-transaction dataset (not named in the part read) | NOT STATED | NOT STATED | VERIFIED (arXiv record); preprint, not peer reviewed |

Content columns come from the abstract (Semantic Scholar / arXiv) or from the team's own review (`docs/Literature_Review_Fraud_Detection.docx`), as named in `tables/literature_survey.csv` -> `summary_source`. **UNVERIFIED items:** the base paper's reported numbers (paywall); Platt (1999/2000, year conflicts between sources; no DOI); see `references.md`.

## 5. Dataset

- **Source:** Kaggle kartik2112/fraud-detection (Sparkov simulated card transactions); loaded with `kagglehub` (`fraudlens/fraudkit/data.py:KAGGLE_DATASET`). Page: https://www.kaggle.com/datasets/kartik2112/fraud-detection (title verified; reachable). The generator is the Sparkov data generation tool (https://github.com/namebrandon/Sparkov_Data_Generation, reachable). **Licence: NOT FOUND** (not checked on the Kaggle page).
- **Files and rows (counted from the cached CSVs):** `fraudTrain.csv` 1,296,675 rows, `fraudTest.csv` 555,719 rows. The provided split is by time and is kept.
- **Fraud rate:** train file 0.579% (7,506 frauds in 1,296,675 rows, 983 cards); test file 0.386% (2,145 frauds in 555,719 rows, 924 cards).
- **Period:** 2019-01-01 00:00:18 to 2020-06-21 12:13:37 (train file); 2020-06-21 12:14:25 to 2020-12-31 23:59:34 (test file).
- **Currency:** none in the data; the app shows amounts as EUR (`fraudlens/configs/default.yaml` comment, Model report data card).
- **Personal data:** `clean()` drops first, last, street, zip and unix_time on load (`fraudlens/fraudkit/data.py`, `clean`); the data are synthetic.

### 5.1 Split sizes (what the model actually uses)
| Part | Rows | Frauds | Fraud rate | Start | End |
|---|---|---|---|---|---|
| fit | 1,102,173 | 6,373 | 0.578% | 2019-01-01 00:00:18 | 2020-04-03 17:53:52 |
| validation | 194,502 | 1,133 | 0.583% | 2020-04-03 17:54:44 | 2020-06-21 12:13:37 |
| test | 555,719 | 2,145 | 0.386% | 2020-06-21 12:14:25 | 2020-12-31 23:59:34 |
| raw_train_file (fit+validation) | 1,296,675 | 7,506 | 0.579% | 2019-01-01 00:00:18 | 2020-06-21 12:13:37 |

*Source:* `fraudlens/artifacts/metrics.json` -> split, data; `tables/dataset_splits.csv`

The validation period is the last 15% of the training file by time (`configs/default.yaml` data.validation_fraction; `fraudkit/train.py:349-358`).

### 5.2 Columns of the raw files
| Column | dtype in CSV | Meaning (inferred; no official data dictionary was found) | Used by FraudLens | Nulls (train) | Distinct (train) |
|---|---|---|---|---|---|
| Unnamed: 0 | int64 | row index of the CSV (dropped by the loader) | dropped | 0 | 1296675 |
| trans_date_trans_time | str | transaction timestamp | yes (time features) | 0 | 1274791 |
| cc_num | int64 | synthetic card number (card identity for history features) | yes (grouping key only) | 0 | 983 |
| merchant | str | merchant name (the prefix 'fraud_' is stripped by the loader) | yes (merchant_te) | 0 | 693 |
| category | str | merchant category | yes | 0 | 14 |
| amt | float64 | transaction amount | yes | 0 | 52928 |
| first | str | cardholder first name | dropped (personal data) | 0 | 352 |
| last | str | cardholder last name | dropped (personal data) | 0 | 481 |
| gender | str | cardholder gender | yes | 0 | 2 |
| street | str | cardholder street | dropped (personal data) | 0 | 983 |
| city | str | cardholder home city | context only | 0 | 894 |
| state | str | cardholder home state | context only | 0 | 51 |
| zip | int64 | cardholder zip | dropped | 0 | 970 |
| lat | float64 | home latitude | yes (dist_home_km) | 0 | 968 |
| long | float64 | home longitude | yes (dist_home_km) | 0 | 969 |
| city_pop | int64 | population of the home city | yes (log_city_pop) | 0 | 879 |
| job | str | cardholder job title | context only | 0 | 494 |
| dob | str | date of birth | yes (age) | 0 | 968 |
| trans_num | str | transaction id | key only | 0 | 1296675 |
| unix_time | int64 | timestamp as unix seconds | dropped | 0 | 1274823 |
| merch_lat | float64 | merchant latitude | yes (distance, speed) | 0 | 1247805 |
| merch_long | float64 | merchant longitude | yes (distance, speed) | 0 | 1275745 |
| is_fraud | int64 | label: 1 = fraud | target | 0 | 2 |

*Source:* command: `report_kit/scripts/extra_facts.py` reading the cached `fraudTrain.csv`; `tables/dataset_columns_raw.csv`


### 5.3 Facts about the training file that explain model behaviour
- Fraud rate in night hours (22:00-03:59) 2.09% vs 0.12% in other hours.
- Largest fraud amount 1,376.04 (median 396.50); largest genuine amount 28,948.90; amounts above 2,000: 731 rows, 0 frauds. A tree model cannot extrapolate beyond this, which motivated the policy rules.
- 908 of the 924 test cards also appear in the training file (983 cards), so the test does not measure performance on unseen cardholders (command: `card_overlap()` in `report_kit/scripts/facts_common.py`, reading the cached CSVs).

Fraud rate by category (training file):

| Category | Rows | Frauds | Fraud rate |
|---|---|---|---|
| shopping_net | 97,543 | 1,713 | 1.76% |
| misc_net | 63,287 | 915 | 1.45% |
| grocery_pos | 123,638 | 1,743 | 1.41% |
| shopping_pos | 116,672 | 843 | 0.72% |
| gas_transport | 131,659 | 618 | 0.47% |
| misc_pos | 79,655 | 250 | 0.31% |
| grocery_net | 45,452 | 134 | 0.29% |
| travel | 40,507 | 116 | 0.29% |
| entertainment | 94,014 | 233 | 0.25% |
| personal_care | 90,758 | 220 | 0.24% |
| kids_pets | 113,035 | 239 | 0.21% |
| food_dining | 91,461 | 151 | 0.17% |
| home | 123,115 | 198 | 0.16% |
| health_fitness | 85,879 | 133 | 0.15% |

## 6. Architecture

Figures: `figures/architecture.png` (system), `figures/scoring_flow.png` (one API request), `figures/sqlite_er.png` (database), `figures/training_pipeline_time_split.png`, `figures/role_matrix.png`. Sources of the diagrams are in `figures/diagrams_src/` (Graphviz source files with the extension dot, and `make_matplotlib_figures.py`).

### 6.1 Python modules (`fraudkit/` plus the API)
| File | Lines | Purpose (first line of the module docstring) |
|---|---|---|
| fraudlens/fraudkit/admin.py | 88 | Operator CLI for users and API keys (for servers and containers without a browser). |
| fraudlens/fraudkit/auth.py | 310 | Users, roles and API keys. |
| fraudlens/fraudkit/bootstrap.py | 58 | Prepare a fresh hosted container before the app starts: fetch the artifacts and create the accounts. |
| fraudlens/fraudkit/cases.py | 198 | Analyst case queue (SQLite) with assignment and a full audit trail. |
| fraudlens/fraudkit/config.py | 131 | Typed, validated configuration (configs/default.yaml). A bad value fails at start-up with a clear message. |
| fraudlens/fraudkit/costs.py | 107 | Money-aware evaluation: cost of missed fraud plus analyst review cost. |
| fraudlens/fraudkit/data.py | 196 | Data loading for FraudLens. |
| fraudlens/fraudkit/db.py | 162 | SQLite storage for users, API keys, cases and audit trails. |
| fraudlens/fraudkit/decision.py | 97 | Decision policy: turn a calibrated probability (plus policy rules) into an action. |
| fraudlens/fraudkit/explain.py | 464 | Per-transaction explanations: SHAP contributions -> plain-English reason codes. |
| fraudlens/fraudkit/features.py | 347 | Leakage-free feature pipeline. |
| fraudlens/fraudkit/monitoring.py | 101 | Data and score drift monitoring with the Population Stability Index (PSI). |
| fraudlens/fraudkit/rules.py | 298 | Policy rules: a guardrail layer on top of the ML score. |
| fraudlens/fraudkit/score.py | 173 | Model bundle: features -> raw score -> calibrated probability -> risk band. |
| fraudlens/fraudkit/security.py | 69 | Small security helpers shared by the app and the API (no Streamlit imports). |
| fraudlens/fraudkit/train.py | 918 | Train FraudLens end to end with a live terminal view. |
| fraudlens/fraudkit/watchlist.py | 88 | Watchlists: blocked cards (stored as SHA-256 hashes) and watched merchants. |
| fraudlens/api/main.py | 336 | FraudLens scoring API. |

*Source:* command: `ast.get_docstring` and line count over the files listed (`report_kit/scripts/facts_part1.py`)


### 6.2 Streamlit pages (`fraudlens/app/common.py` PAGES; entry point `app/streamlit_app.py`)
| Section | Page | File | Permission | Roles that can open it |
|---|---|---|---|---|
| Overview | Dashboard | fraudlens/app/views/1_Dashboard.py | view | viewer, analyst, admin |
| Overview | Monitoring | fraudlens/app/views/7_Monitoring.py | view | viewer, analyst, admin |
| Investigate | Case queue | fraudlens/app/views/4_Case_Queue.py | view | viewer, analyst, admin |
| Investigate | Check a transaction | fraudlens/app/views/2_Check_Transaction.py | view | viewer, analyst, admin |
| Investigate | Card 360 | fraudlens/app/views/10_Card_360.py | view | viewer, analyst, admin |
| Investigate | Batch scoring | fraudlens/app/views/3_Batch_Scoring.py | score | analyst, admin |
| Model | Model report | fraudlens/app/views/5_Model_Report.py | view | viewer, analyst, admin |
| Model | Training Lab | fraudlens/app/views/6_Training_Lab.py | view | viewer, analyst, admin |
| Settings | Admin | fraudlens/app/views/8_Admin.py | manage_users | admin |
| Settings | My account | fraudlens/app/views/9_Account.py | view | viewer, analyst, admin |

Inside a page, extra rules apply (e.g. Check a transaction: typed input and what-if need `score`; Case queue: acting needs `decide_cases`, clearing needs `clear_queue`; Training Lab: Retrain needs `retrain`). Source: `fraudlens/app/views/*.py` (`allowed(...)` calls).

### 6.3 API endpoints (`fraudlens/api/main.py`)
| Method | Path | Auth | Purpose | Source |
|---|---|---|---|---|
| GET | /v1/health | none | liveness probe | api/main.py:235 |
| GET | /v1/ready | none | model loaded + model version | api/main.py:240 |
| GET | /v1/model | X-API-Key | model version, threshold, metrics summary | api/main.py:250 |
| POST | /v1/score | X-API-Key | score up to 1,000 transactions; returns probability, band, decision, risk score, reasons | api/main.py:270 |
| GET | /health | none | legacy route kept for old clients | api/main.py:329 |
| POST | /score | X-API-Key | legacy route (returns a bare list) | api/main.py:334 |

Limits: 1000 transactions per request (HTTP 413 above), 120 requests per minute per key (HTTP 429), field validation errors return HTTP 422; every response carries `request_id` and `model_version`. Source: `configs/default.yaml` api; `api/main.py:270-327`; `tests/test_api.py` (8 tests).

### 6.4 SQLite schema (`fraudlens/fraudkit/db.py` MIGRATIONS, schema version 3)
Tables: `cases`, `audit`, `users`, `api_keys`, `admin_audit`, `watchlist`, `case_comments` (columns and foreign keys in `figures/sqlite_er.png`; DDL in `fraudkit/db.py:14-107`). WAL journal mode and `busy_timeout` are set in `connect()` (`db.py:114-120`). Migrations are applied in one transaction each and tracked with `PRAGMA user_version` (`db.py:123-135`); `tests/test_platform.py::test_migration_upgrades_old_case_database` upgrades an old database in place.

### 6.5 Authentication and roles (`fraudlens/fraudkit/auth.py`)
| Permission | viewer | analyst | admin |
|---|---|---|---|
| view | True | True | True |
| score | False | True | True |
| decide_cases | False | True | True |
| feed_queue | False | True | True |
| clear_queue | False | False | True |
| retrain | False | False | True |
| manage_users | False | False | True |
| manage_api_keys | False | False | True |
| view_audit | False | False | True |

Passwords: PBKDF2-HMAC-SHA256, 600,000 iterations, random 16-byte salt, minimum length 10 with a letter and a digit, not containing the username (`auth.py:35-36,49-72`). Lockout after 5 failed attempts for 15 minutes and a 60-minute session timeout (`configs/default.yaml` app). API keys: random `fl_...` token shown once, stored as a SHA-256 hash, revocable, with use counts (`auth.py:248-268`). The OWASP Password Storage Cheat Sheet lists "PBKDF2-HMAC-SHA256: 600,000 iterations" (page fetched during this kit's verification).

### 6.6 Configuration (`fraudlens/configs/default.yaml`, validated by pydantic in `fraudkit/config.py`)
| Key | Value |
|---|---|
| seed | 42 |
| data.cache_dir |  |
| data.validation_fraction | 0.15 |
| features.merchant_encoding | target |
| features.te_smoothing | 50 |
| model.optuna_trials | 30 |
| model.cv_folds | 3 |
| model.tuning_sample_negatives | 0.25 |
| costs.review_cost | 5.0 |
| bands.medium |  |
| bands.high | 0.3 |
| bands.critical | 0.7 |
| explain.top_raising | 4 |
| explain.top_lowering | 2 |
| decision.mode | auto |
| decision.sla_hours | 24 |
| rules.enabled | True |
| rules.review_amount | auto |
| rules.spend_spike_ratio | auto |
| rules.spend_spike_min_amount | auto |
| rules.far_from_home_km | auto |
| rules.burst_txn_1h | 5 |
| rules.out_of_range | True |
| rules.block_confirmed_cards | True |
| app.db | data/fraudlens.sqlite |
| app.session_timeout_minutes | 60 |
| app.max_failed_logins | 5 |
| app.lockout_minutes | 15 |
| app.max_upload_rows | 20000 |
| api.max_batch | 1000 |
| api.rate_limit_per_minute | 120 |

### 6.7 Deployment
- **Local Docker:** `fraudlens/Dockerfile` (python:3.11-slim, non-root user, health check) and `fraudlens/docker-compose.yml` (api on 127.0.0.1:8000, app on 127.0.0.1:8501, artifacts mounted read-only, data in a named volume).
- **Hugging Face Space (prepared, NOT deployed):** `fraudlens/deploy/space/Dockerfile`, `fraudlens/fraudkit/bootstrap.py` (refuses to start without admin and demo passwords, downloads artifacts, creates accounts) and `fraudlens/scripts/deploy_space.py`. The Space image was built and run locally during development and served sign-in, the dashboard and a 500-row batch (`fraudlens/PROGRESS.md` M12); no public URL exists yet.
- **CI:** `.github/workflows/fraudlens-ci.yml` (section 9).

## 7. Methodology

### 7.1 Features (28 model inputs) and leakage prevention
Rule (docstring of `fraudkit/features.py:1-7`): every card-behaviour feature of a transaction is computed **only from strictly earlier transactions of the same card**. Rows are sorted by (card, time, original position) so ties are resolved deterministically (`features.py:153-155`); the rolling-window helper counts rows `j < i` with `t_j >= t_i - window` (`features.py:92-99`). Merchant and category risk use labels only if they are at least 7 days old (`features.py:17,128-135`). Merchant target encoding is out-of-fold and fitted on the fit split only (`features.py:284-320`).
Tests (`fraudlens/tests/test_features.py`, 7 tests; names copied from the file): `test_no_feature_uses_future_rows`, `test_future_perturbation_does_not_change_past`, `test_featurize_with_history_matches_batch`, `test_velocity_and_first_time_semantics`, `test_haversine_known_distance`, `test_merchant_encoder_oof_does_not_use_own_label`, `test_delayed_risk_ignores_recent_labels` (run results in section 9).

| Feature | Group | Definition / formula | Code | Mean abs SHAP (log-odds) | Rank |
|---|---|---|---|---|---|
| amt | raw | Transaction amount (currency units of the dataset, shown as EUR) | column amt | 0.9573 | 1 |
| log_amt | amount | log(1 + amt) | fraudlens/fraudkit/features.py:159 | 0.2028 | 14 |
| hour | time | Hour of day of the transaction (0-23) | fraudlens/fraudkit/features.py:160 | 0.4492 | 8 |
| day_of_week | time | Day of week (Monday = 0) | fraudlens/fraudkit/features.py:161 | 0.1912 | 15 |
| is_night | time | 1 if hour is in {22, 23, 0, 1, 2, 3} | fraudlens/fraudkit/features.py:14,162 | 0.6683 | 3 |
| dist_home_km | geo | Haversine distance (km) between the cardholder's home (lat, long) and the merchant (merch_lat, merch_long) | fraudlens/fraudkit/features.py:81-84,163 | 0.1005 | 23 |
| age | demographic | (transaction time - date of birth) in days / 365.25 | fraudlens/fraudkit/features.py:164 | 0.4442 | 9 |
| log_city_pop | demographic | log(1 + city_pop) | fraudlens/fraudkit/features.py:165 | 0.4862 | 5 |
| merchant_te | merchant | Smoothed target encoding of merchant: (sum_fraud + m * prior) / (count + m), m = 50; out-of-fold (5 folds) for fit rows, fitted on the fit split only | fraudlens/fraudkit/features.py:284-320 | 0.18 | 16 |
| n_txn_1h | velocity | Number of EARLIER transactions of the card in the previous 1 hour | fraudlens/fraudkit/features.py:92-99,209-212,237 | 0.0279 | 26 |
| n_txn_24h | velocity | Earlier transactions of the card in the previous 24 hours | fraudlens/fraudkit/features.py:211,238 | 0.1681 | 17 |
| amt_sum_24h | velocity | Sum of amounts of the card's earlier transactions in the previous 24 hours | fraudlens/fraudkit/features.py:211,240 | 0.3672 | 11 |
| n_txn_7d | velocity | Earlier transactions of the card in the previous 7 days | fraudlens/fraudkit/features.py:250,255 | 0.1554 | 19 |
| n_txn_30d | velocity | Earlier transactions of the card in the previous 30 days (replaced the lifetime counter card_txn_count, which drifted) | fraudlens/fraudkit/features.py:212,239 | 0.4728 | 6 |
| avg_amt_1d | spending window | Mean amount of the card's earlier transactions in the last 1 day (NaN if none) | fraudlens/fraudkit/features.py:256 | 0.5244 | 4 |
| avg_amt_7d | spending window | Same, 7 days | fraudlens/fraudkit/features.py:257 | 0.2229 | 13 |
| avg_amt_30d | spending window | Same, 30 days | fraudlens/fraudkit/features.py:258 | 0.1674 | 18 |
| amt_to_avg_30d | spending window | amt / avg_amt_30d | fraudlens/fraudkit/features.py:259 | 0.0927 | 24 |
| amt_to_card_mean | card profile | amt / mean of all the card's earlier amounts | fraudlens/fraudkit/features.py:169-174 | 0.1153 | 22 |
| amt_to_cat_mean | card profile | amt / mean of the card's earlier amounts in the same category | fraudlens/fraudkit/features.py:176-182 | 0.3692 | 10 |
| amt_log_z | card profile | (log(1+amt) - mean) / std of log(1+amount) over the card's earlier transactions (needs >= 2 earlier rows and std > 1e-6) | fraudlens/fraudkit/features.py:260-267 | 0.1501 | 20 |
| first_time_category | card profile | 1 if this is the card's first purchase in this category | fraudlens/fraudkit/features.py:184 | 0.0001 | 27 |
| hours_since_prev | sequence | Hours since the card's previous transaction | fraudlens/fraudkit/features.py:187-189 | 0.451 | 7 |
| geo_speed_kmh | geo | Distance between the previous and the current merchant location / max(hours since previous, 1/60 h) | fraudlens/fraudkit/features.py:191-196 | 0.1499 | 21 |
| impossible_travel | geo | 1 if geo_speed_kmh > 800 and the jump > 50 km | fraudlens/fraudkit/features.py:15,198 | 0.0 | 28 |
| small_txn_24h | card testing | Number of the card's earlier transactions below 5 in the last 24 hours | fraudlens/fraudkit/features.py:16,247,252,268 | 0.2564 | 12 |
| category | categorical | Merchant category (14 levels in the training file) | column category | 0.8903 | 2 |
| gender | categorical | Cardholder gender (see ethics note) | column gender | 0.0604 | 25 |

*Source:* `fraudlens/fraudkit/features.py` (line numbers in the Code column); `fraudlens/artifacts/metrics.json` -> global_importance (5,000 random validation rows); `tables/features_catalog.csv`

Context columns (not model inputs): prev_city_km, minutes_since_prev, card_mean_amt, cat_mean_amt, card_txn_count, card_max_amt, amt_to_card_max, cat_share, hour_share (`features.py:68-78`). `card_txn_count` was removed from the model because its PSI was 4.19 on the test period (section 8.4).

### 7.2 Models, tuning, final training
- Baselines trained on all fit rows (`fraudkit/train.py:397-447`): logistic regression (log1p + median imputation + standard scaling + one-hot, `C=1.0`, `max_iter=2000`); random forest (200 trees, `min_samples_leaf=5`, `max_features="sqrt"`, `max_samples=0.3`); LightGBM with default parameters. **None of the baselines is tuned**, only LightGBM is.
- Tuned LightGBM: Optuna TPE sampler (seed 42), 30 trials, 3-fold stratified shuffled CV on the fit period, objective mean PR-AUC, 25% of genuine rows kept per fold (`train.py:453-477`). Search ranges: n_estimators 150-1200 (step 50), learning_rate 0.01-0.2 (log), num_leaves 15-255 (log), min_child_samples 10-300 (log), subsample 0.5-1.0, colsample_bytree 0.4-1.0, reg_lambda and reg_alpha 1e-3-10 (log), scale_pos_weight 1-10 (log). Best CV PR-AUC 0.9866.
- The final model is refit with the best parameters on the fit rows with the same 25% genuine subsample (`train.py:516-520`), then scored on validation.

Best parameters (`metrics.json` tuning.best_params):

| Parameter | Value |
|---|---|
| n_estimators | 900 |
| learning_rate | 0.054954 |
| num_leaves | 23 |
| min_child_samples | 111 |
| subsample | 0.821601 |
| colsample_bytree | 0.815616 |
| reg_lambda | 0.0520622 |
| reg_alpha | 0.0129065 |
| scale_pos_weight | 4.78075 |

Validation leaderboard (last 15% of the training file, 194,502 rows, 1,133 frauds):

| Model | PR-AUC | ROC-AUC | Brier | Fit seconds | Selected |
|---|---|---|---|---|---|
| LightGBM tuned (Optuna) | 0.9872 | 0.9998 | 0.00053 | 5.41 | True |
| Random forest | 0.9570 | 0.9962 | 0.00098 | 35.88 | False |
| LightGBM default | 0.7977 | 0.9580 | 0.00173 | 2.49 | False |
| Logistic regression | 0.7597 | 0.9814 | 0.00231 | 6.83 | False |

### 7.3 Calibration
Down-sampling genuine rows and weighting frauds shifts raw scores away from true probabilities, so a calibrator is fitted on the validation period. Three calibrators are compared on two time-halves of validation (fit on one half, score on the other, both ways); the lowest out-of-sample log-loss wins (`train.py` calibration stage, `score.py:42-85`). Output is clipped to [1e-12, 1-1e-12] so a probability is never exactly 0 or 1.

| Method | Log-loss | Brier | Distinct values | Share at floor |
|---|---|---|---|---|
| isotonic | 0.00250 | 0.000382 | 22 | 97.66% |
| platt | 0.00159 | 0.000374 | 190927 | 0.00% |
| beta | 0.00162 | 0.000374 | 193417 | 0.00% |

Chosen: **platt**. Test Brier score 0.00050 before and 0.00037 after calibration; test log-loss 0.00157; 522,809 distinct probabilities after rounding to 12 decimals (555,719 distinct raw values). Reliability table: `tables/reliability_test_final.csv`. Why scores once looked like only 0% or 100%: isotonic regression is a step function (the validation halves show 22 distinct values and 97.66% at the floor above). The "95% of test rows exactly 0 and 94 distinct values" figures in `PROGRESS.md` M11 describe the retired isotonic model whose test scores were not kept: **NOT re-derivable from retained artifacts**.

### 7.4 Risk score 0-999
`risk_score(p)` = round((logit(p) - logit(1e-6)) / (logit(0.999) - logit(1e-6)) x 999), clipped to 0-999 (`score.py:29-39`). About 33.4 points double the odds. On the test set the median score is 24 for genuine and 950 for fraud transactions; the decision threshold p = 0.0837 is score 551, High (p = 0.30) is 625 and Critical (p = 0.70) is 707 (command: `report_kit/scripts/extra_facts.py`). Decisions use the probability; the score is for reading.

### 7.5 Decision logic
- **Threshold chosen on validation by cost:** total cost = amount of frauds not flagged + EUR 5 x flagged transactions; minimum at p = 0.0837 (`metrics.json` threshold; `fraudkit/costs.py`).
- **Two policies compared on validation** (`fraudkit/decision.py`): flat threshold (flag when p >= t) vs expected loss (flag when p x amount >= k x review cost; k from the grid 0.25-8, here k = 8, the largest grid value). The cheaper one on validation is served: **threshold**.
- **Actions:** block if band is Critical (p >= 0.70 or a Critical rule), review if the policy flags it or a High rule fired, step-up if band is Medium or higher or a Medium rule fired, otherwise approve (`decision.py:43-52`).
- Bands: Medium starts at the decision threshold, High at 0.30, Critical at 0.70 (`configs/default.yaml` bands).

| policy | split | flagged | tp | fp | fn | precision | recall | total_cost | saved_vs_no_model |
|---|---|---|---|---|---|---|---|---|---|
| threshold | validation | 1274 | 1109 | 165 | 24 | 0.8705 | 0.9788 | 8,569.66 | 595,538.82 |
| threshold | test | 2463 | 2057 | 406 | 88 | 0.8352 | 0.9590 | 30,137.00 | 1,103,187.68 |
| expected_loss | validation | 1189 | 1078 | 111 | 55 | 0.9066 | 0.9515 | 8,948.46 | 595,160.02 |
| expected_loss | test | 2278 | 1982 | 296 | 163 | 0.8701 | 0.9240 | 28,475.99 | 1,104,848.69 |

### 7.6 Policy rules (guardrail; can only raise the band)
Added after a EUR 12,000 purchase 2,500 km from home scored Low (0.01%): in the training data no fraud exceeds EUR 1,376.04 and amounts above 2,000 are never fraud, so the tree cannot extrapolate (`fraudkit/rules.py:1-8`; screenshot `2c_policy_rules.png`). Limits are derived from training data (`metrics.json` rules_auto):
- large amount >= 2,000 -> High (1.25 x largest fraud amount in the fit period (1,371.81), rounded up to 500);
- far from home >= 400 km -> Medium (2 x largest home-merchant distance in the fit period, rounded up to 100 km);
- spending spike >= 50x the card's average and amount >= 2,000 -> High (grid search on validation, lowest total cost of policy + rule (ties -> stricter));
- impossible travel (speed flag and a jump >= far-from-home limit) -> High; burst of >= 5 other transactions in 1 hour -> High; any key feature outside the training range -> Medium; blocked card -> Critical; watched merchant -> High (`rules.py:87-122`, `configs/default.yaml` rules).
Measured effect on the test set with the final model:

| system | flagged | tp | fp | fn | precision | recall | false_alarms_per_10k_genuine | review_cost_total | total_cost |
|---|---|---|---|---|---|---|---|---|---|
| Model policy alone | 2463 | 2057 | 406 | 88 | 0.8352 | 0.9590 | 7.33 | 12,315 | 30,137 |
| Model policy + rules | 2871 | 2057 | 814 | 88 | 0.7165 | 0.9590 | 14.70 | 14,355 | 32,177 |

| rule | band | effect | hits | frauds_in_hits | extra_reviews | extra_frauds_caught |
|---|---|---|---|---|---|---|
| large_amount | High | review | 359 | 0 | 347 | 0 |
| spend_spike | High | review | 169 | 0 | 165 | 0 |
| far_from_home | Medium | step-up | 0 | 0 | 0 | 0 |
| impossible_travel | High | review | 0 | 0 | 0 | 0 |
| burst | High | review | 70 | 9 | 61 | 0 |
| out_of_range | Medium | step-up | 4 | 2 | 0 | 0 |

On this simulated test set the rules add reviews and catch no extra fraud; their purpose is protection outside the training data. Source: `tables/policy_rules_*.csv` (command: `fraudkit.rules.policy_impact`, run by `report_kit/scripts/extra_facts.py`).

### 7.7 SHAP reason codes: three real examples
SHAP `TreeExplainer` gives each feature's contribution in log-odds; related features are grouped; groups at or above 0.05 log-odds become reasons (top 4 raising, top 2 lowering by default); wording is filled with the transaction's real values (`fraudkit/explain.py:334-376`). SHAP values plus the base value reconstruct the model's raw margin (`tests/test_explain.py`, tolerance 1e-6). The three examples below were produced by calling `explain_frame` on rows of `artifacts/test_scored.parquet` with the served model and policy (`report_kit/scripts/extract_results.py`; raw output `_data/shap_examples.json`):


**caught fraud (true positive, highest p)** (`trans_num` cb9b6ae471a8f51bdc2a640fd905d512; dataset label is_fraud = 1; amount 856.30; category misc_net; hour 22)
- calibrated probability 0.999993 (stored value 0.999993); risk score 999; band Critical; decision **block**; expected loss 856.29
- summary text shown to the analyst: "Critical risk (>99.99%) mainly because large amount of €856.30, plus 3 more signals."
- raises risk (+9.35 log-odds): Large amount of €856.30
- raises risk (+7.55 log-odds): Amount is 5.4x this card's 30-day average (€157.78; 11 transactions in 7 days)
- raises risk (+4.42 log-odds): 3 other transaction(s) on this card in the last hour, 6 in the last 24 hours (€3,060.63)
- raises risk (+4.27 log-odds): Transaction at 10:57 PM (night-time)
- lowers risk (-0.22 log-odds): Home city population 5,732
- lowers risk (-0.14 log-odds): Category 'misc_net' is rarely involved in fraud
- policy rules fired: 0

**false alarm (genuine flagged, highest p)** (`trans_num` def046458db77bdd7a63d7edac50311d; dataset label is_fraud = 0; amount 774.23; category misc_net; hour 1)
- calibrated probability 0.988946 (stored value 0.988946); risk score 883; band Critical; decision **block**; expected loss 765.67
- summary text shown to the analyst: "Critical risk (98.9%) mainly because large amount of €774.23, plus 3 more signals."
- raises risk (+9.26 log-odds): Large amount of €774.23
- raises risk (+3.86 log-odds): Transaction at 1:53 AM (night-time)
- raises risk (+2.40 log-odds): Amount is 22.3x what this card usually spends on 'misc_net' (average €34.75)
- raises risk (+1.80 log-odds): Amount is 13.7x this card's 30-day average (€56.67; 18 transactions in 7 days)
- lowers risk (-0.88 log-odds): Category 'misc_net' is rarely involved in fraud
- lowers risk (-0.26 log-odds): Location is consistent with the previous transaction (144 km in 69 minutes)
- policy rules fired: 0

**missed fraud (false negative, highest p below threshold)** (`trans_num` 4d069e0e767b3163f62736966d64f6ed; dataset label is_fraud = 1; amount 772.07; category misc_net; hour 7)
- calibrated probability 0.079921 (stored value 0.079921); risk score 548; band Low; decision **approve**; expected loss 61.70
- summary text shown to the analyst: "Low risk (8.0%). The strongest risk signal is large amount of €772.07, but it is outweighed by: category 'misc_net' is rarely involved in fraud."
- raises risk (+8.70 log-odds): Large amount of €772.07
- raises risk (+1.79 log-odds): Amount is 2.6 standard deviations above this card's usual spending
- raises risk (+1.67 log-odds): Amount is 10.5x what this card usually spends on 'misc_net' (average €73.77)
- raises risk (+1.04 log-odds): Cardholder age 27 (higher-risk age group in the data)
- lowers risk (-1.18 log-odds): Category 'misc_net' is rarely involved in fraud
- lowers risk (-0.77 log-odds): Home city population 116,001
- policy rules fired: 0

### 7.8 Drift monitoring (PSI)
For every feature and the score, the reference profile stores 10 quantile bins of the fit period (`fraudkit/monitoring.py:53-76`); new data is binned with the same edges and PSI = sum((actual - expected) x ln(actual / expected)) with shares clipped at 1e-4. Reading: < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant (`monitoring.py:13-14,39-50`). Results are in section 8.4.

## 8. Results

All numbers below are read from `fraudlens/artifacts/metrics.json` (final run) and `fraudlens/artifacts/runs/*/metrics.json` (archive of the six runs); tables are also saved in `tables/`. **Evaluation protocol:** fit on the oldest 85% of the training file, tune with CV, calibrate and choose the threshold on the last 15% (validation), evaluate **once** on the later test file. Review cost EUR 5 per flagged transaction.

### 8.1 Final model: test results (model `20261026-184402-d37e9778`)
| Metric | Value | Source |
|---|---|---|
| Test transactions / frauds | 555,719 / 2,145 | metrics.json split.test |
| PR-AUC (95% bootstrap CI, 1000 resamples) | 0.9765 (0.9719 - 0.9811) | test.pr_auc, pr_auc_ci95 |
| ROC-AUC | 0.9995 | test.roc_auc |
| Brier score before / after calibration | 0.00050 / 0.00037 | test.brier_* |
| Decision threshold (chosen on validation) | 0.0837 | threshold.value |
| Flagged (review or block) | 2,463 | test.at_threshold.flagged |
| True / false positives, false negatives | 2,057 / 406, 88 | test.at_threshold |
| Precision / recall | 83.52% / 95.90% | test.at_threshold |
| False alarms per 10,000 genuine | 7.33 | test.at_threshold |
| Fraud amount in the test period | 1,133,324.68 | test.at_threshold.fraud_amount |
| Caught / missed amount | 1,115,502.68 / 17,822.00 | test.at_threshold |
| Review cost / total cost (missed + reviews) | 12,315 / 30,137.00 | test.at_threshold |
| Saved vs "no model" | 1,103,187.68 | test.at_threshold.saved_vs_no_model |
| Training wall-clock of this run | 68,988.9 s (includes the computer sleeping; the identical-metrics archived run took 1,236.2 s) | run.total_seconds; runs/20261025-233115-3abe0ebd |

The validation leaderboard, calibration comparison and decision-policy comparison are in section 7. The reliability table (observed vs predicted fraud rate in fixed probability bins) is `tables/reliability_test_final.csv` (24 rows).

### 8.2 Every archived run (quality)
| Model version | Features | Calibration | Test PR-AUC | CI low | CI high | ROC-AUC | Brier (calibrated) | Precision | Recall | False alarms / 10k |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261020-153708-3c73d465 | 21 | isotonic | 0.9682 | 0.9620 | 0.9743 | 0.9978 | 0.00040 | 0.8552 | 0.9557 | 6.27 |
| 20261020-154750-a6506be0 | 21 | isotonic | 0.9722 | 0.9666 | 0.9773 | 0.9978 | 0.00039 | 0.8112 | 0.9636 | 8.69 |
| 20261020-172652-a6506be0 | 21 | isotonic | 0.9722 | 0.9666 | 0.9773 | 0.9978 | 0.00039 | 0.8112 | 0.9636 | 8.69 |
| 20261025-224238-a6506be0 | 21 | isotonic | 0.9722 | 0.9666 | 0.9773 | 0.9978 | 0.00039 | 0.8112 | 0.9636 | 8.69 |
| 20261025-233115-3abe0ebd | 28 | platt | 0.9765 | 0.9719 | 0.9811 | 0.9995 | 0.00037 | 0.8352 | 0.9590 | 7.33 |
| 20261026-184402-d37e9778 | 28 | platt | 0.9765 | 0.9719 | 0.9811 | 0.9995 | 0.00037 | 0.8352 | 0.9590 | 7.33 |

### 8.3 Every archived run (money and time)
| Model version | Wall-clock s | Tuning s | Threshold | Flagged | TP | FP | FN | Caught EUR | Missed EUR | Review cost EUR | Total cost EUR | Saved vs no model EUR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261020-153708-3c73d465 | 562.7 | 508.0 | 0.1111 | 2,397 | 2050 | 347 | 95 | 1,114,206 | 19,119 | 11,985 | 31,104 | 1,102,221 |
| 20261020-154750-a6506be0 | 502.0 | 446.7 | 0.0238 | 2,548 | 2067 | 481 | 78 | 1,117,331 | 15,994 | 12,740 | 28,734 | 1,104,591 |
| 20261020-172652-a6506be0 | 1,235.3 | 1,135.6 | 0.0238 | 2,548 | 2067 | 481 | 78 | 1,117,331 | 15,994 | 12,740 | 28,734 | 1,104,591 |
| 20261025-224238-a6506be0 | 672.2 | 604.2 | 0.0238 | 2,548 | 2067 | 481 | 78 | 1,117,331 | 15,994 | 12,740 | 28,734 | 1,104,591 |
| 20261025-233115-3abe0ebd | 1,236.2 | 1,079.9 | 0.0837 | 2,463 | 2057 | 406 | 88 | 1,115,503 | 17,822 | 12,315 | 30,137 | 1,103,188 |
| 20261026-184402-d37e9778 | 68,988.9 | 68,884.8 | 0.0837 | 2,463 | 2057 | 406 | 88 | 1,115,503 | 17,822 | 12,315 | 30,137 | 1,103,188 |

*Source:* `tables/runs_comparison.csv` (generated by `report_kit/scripts/extract_results.py` from `fraudlens/artifacts/runs/*/metrics.json`)

Validation PR-AUC of each model in each run: `tables/leaderboard_by_run.csv`. Optuna history of the final run: `tables/optuna_trials_final.csv` (best CV PR-AUC 0.9866, median 0.9833, worst 0.9664, first trial 0.9757).

**Reading the run table.** Runs 2, 3 and 4 (`...154750`, `...172652`, `20261025-224238`) share the model-hash suffix `a6506be0` and identical test metrics: the same model was retrained to add stored artifacts (training ranges for the rules, decision data), with wall-clock times of 502, 1,235 and 672 s. Runs 5 and 6 have identical test metrics but different hash suffixes (INCONSISTENCIES I7).

### 8.4 Drift findings (PSI of the served model's inputs on the test period vs the fit-period reference)
Whole test period, top features (all stable, PSI < 0.10):

| Feature | Kind | PSI | Status |
|---|---|---|---|
| merchant_te | numeric | 0.0880 | stable |
| n_txn_30d | numeric | 0.0878 | stable |
| n_txn_7d | numeric | 0.0862 | stable |
| day_of_week | numeric | 0.0737 | stable |
| amt_to_cat_mean | numeric | 0.0561 | stable |
| n_txn_24h | numeric | 0.0455 | stable |
| amt_sum_24h | numeric | 0.0289 | stable |
| hours_since_prev | numeric | 0.0182 | stable |

By month (top feature, number of moderate and significant features, PSI of the score):

| Month | Rows | Top feature | Top PSI | Moderate | Significant | Score PSI |
|---|---|---|---|---|---|---|
| 2020-06 | 30,058 | day_of_week | 0.2699 | 1 | 1 | 0.0008 |
| 2020-07 | 85,848 | n_txn_30d | 0.1607 | 1 | 0 | 0.0013 |
| 2020-08 | 88,759 | n_txn_30d | 0.1467 | 1 | 0 | 0.0005 |
| 2020-09 | 69,533 | day_of_week | 0.1193 | 1 | 0 | 0.0003 |
| 2020-10 | 69,348 | n_txn_30d | 0.1239 | 2 | 0 | 0.0003 |
| 2020-11 | 72,635 | n_txn_30d | 0.1097 | 2 | 0 | 0.0008 |
| 2020-12 | 139,538 | n_txn_7d | 1.0304 | 5 | 4 | 0.0068 |

December 2020, features above 0.10:

| Feature | PSI | Status |
|---|---|---|
| n_txn_7d | 1.0304 | significant |
| n_txn_24h | 0.5871 | significant |
| n_txn_30d | 0.5168 | significant |
| amt_sum_24h | 0.3880 | significant |
| hours_since_prev | 0.2120 | moderate |
| geo_speed_kmh | 0.1581 | moderate |
| small_txn_24h | 0.1508 | moderate |
| avg_amt_1d | 0.1354 | moderate |
| day_of_week | 0.1079 | moderate |

- December has 139,538 test rows against 72,635 in November, so the velocity features (n_txn_7d, n_txn_24h, n_txn_30d, amt_sum_24h) shift. The model score barely moves (PSI 0.007). That December is a seasonal spike is the README's explanation; this kit did NOT test it.
- **The defect found by drift monitoring:** the lifetime counter `card_txn_count` had PSI 4.19 on the whole test period vs the fit period (monthly values 3.71 - 6.11, `tables/drift_card_txn_count_by_month.csv`) and was the 5th most important feature in the earlier model (PROGRESS M8, not re-derivable: that model is no longer stored). It was replaced by `n_txn_30d`.
- `n_txn_30d` itself shows moderate monthly drift (0.11 - 0.16 in July to November): see INCONSISTENCIES I4.

*Source:* `tables/drift_test_period_overall.csv`, `drift_test_period_by_month.csv`, `drift_2020_12_all_features.csv`, `drift_card_txn_count_by_month.csv`; command: `fraudkit.monitoring.drift_report` over `artifacts/test_scored.parquet` and `artifacts/reference_profile.json` (`report_kit/scripts/extract_results.py`, `extra_facts.py`)


### 8.5 Feature ablation (validation PR-AUC, two seeds)
Candidates were added one group at a time to the earlier model's feature set and trained with the tuned parameters; a group is kept only if it beats the baseline by more than twice the seed noise (0.00045). Run on model `20261025-224238-a6506be0` (the model before the research features).

| Candidate | Seed 42 | Seed 7 | Mean | Delta vs baseline | x seed noise |
|---|---|---|---|---|---|
| current feature set | 0.98445 | 0.98490 | 0.98467 | 0.00000 | 0.00 |
| + amount vs card's previous maximum | 0.98485 | 0.98400 | 0.98443 | -0.00024 | -0.54 |
| + share of the card's purchases in this category | 0.98473 | 0.98398 | 0.98436 | -0.00031 | -0.70 |
| + how usual this hour is for the card | 0.98478 | 0.98525 | 0.98502 | 0.00035 | 0.77 |
| + spending windows (RFM) | 0.98560 | 0.98529 | 0.98544 | 0.00077 | 1.70 |
| + amount z-score vs card history | 0.98515 | 0.98491 | 0.98503 | 0.00035 | 0.79 |
| + card-testing probes | 0.98574 | 0.98591 | 0.98583 | 0.00115 | 2.56 |
| + weekend + circular hour | 0.98347 | 0.98428 | 0.98388 | -0.00080 | -1.77 |
| + merchant/category fraud risk (7-day label delay) | 0.98432 | 0.98439 | 0.98435 | -0.00032 | -0.71 |
| + merchant velocity | 0.98428 | 0.98455 | 0.98441 | -0.00026 | -0.58 |
| + all research groups | 0.98647 | 0.98629 | 0.98638 | 0.00171 | 3.78 |
| + all research groups except weekend/hour | 0.98631 | 0.98623 | 0.98627 | 0.00160 | 3.54 |
| + spending windows + z-score + card testing | 0.98619 | 0.98655 | 0.98637 | 0.00170 | 3.77 |

Kept by the rule (`artifacts/feature_ablation.json` -> kept): + card-testing probes, + all research groups, + all research groups except weekend/hour, + spending windows + z-score + card testing. The model uses the simplest equal set (spending windows + amount z-score + card-testing probes, 7 features; `fraudkit/features.py:52-59`). Weekend + circular hour alone hurt (-0.00080). Source: `scripts/feature_ablation.py`, `tables/feature_ablation.csv`.

### 8.6 What improved between runs, and why
| Step | Change | Test PR-AUC | Other measured effects | Evidence for the "why" |
|---|---|---|---|---|
| Run 1 -> 2 | Replaced lifetime counter `card_txn_count` by `n_txn_30d` | 0.9682 -> 0.9722 | recall 95.6% -> 96.4%; precision 85.5% -> 81.1%; saved EUR 1,102,221 -> 1,104,591; threshold 0.1111 -> 0.0238 | PSI 4.19 for the retired feature (section 8.4); PROGRESS M8 |
| Run 2 -> 3 -> 4 | Same model retrained to store rule limits, decision policy data | unchanged (0.9722) | none | identical hash suffix `a6506be0`; PROGRESS M9, M10 |
| Run 4 -> 5/6 | 7 research-based features (28 inputs), Platt instead of isotonic calibration | 0.9722 -> 0.9765 | ROC-AUC 0.9978 -> 0.9995; precision 81.1% -> 83.5%; recall 96.4% -> 95.9%; false alarms/10k 8.7 -> 7.3; threshold 0.0238 -> 0.0837; **total cost EUR 28,734 -> 30,137** (higher by 1,403) | ablation table (section 8.5); calibration table (section 7.3) |

- Overall PR-AUC change run 1 -> final: +0.0082. The 95% CIs of consecutive runs overlap (run 1 0.9620-0.9743; final 0.9719-0.9811), so the improvement is small relative to the uncertainty and was measured on a single test set.
- **The money outcome did not improve in the last step:** total cost went up by EUR 1,403 (saved vs no model 1,104,591 -> 1,103,188), because the validation-chosen threshold moved (0.0238 -> 0.0837) and recall fell. The better ranking (PR-AUC, ROC-AUC) and the smooth probabilities are the gains; do not claim a cost reduction.
- Why the research features were added: scores in the app looked like only 0% or 100% (PROGRESS M11); literature-based features were then tested on validation. Source of the reasons: `fraudlens/PROGRESS.md` M8-M11 (documentation, cross-checked against the numbers above).

### 8.7 Limitations of the evaluation (be honest in the report)
1. **Simulated data** (Sparkov generator): patterns are cleaner than real fraud; the app's own Model report says real-world PR-AUC "would be much lower".
2. **One time split, one test period**; no repeated time-based validation. The 95% CI only reflects resampling of test rows.
3. **Cards overlap:** 908 of the 924 test cards also appear in the training file (983 cards), so the test does not measure unseen cardholders or cold start.
4. **Optuna CV is stratified and shuffled** (`train.py:453`), not time- or card-aware, so the CV score (0.9866) is not a time-honest estimate.
5. **Baselines are untuned** (logistic regression, random forest, default LightGBM) while LightGBM got 30 trials, so the leaderboard gap is partly tuning.
6. **Bootstrap CI treats rows as independent**, although a card's transactions are correlated.
7. **Money metric assumptions** (`fraudkit/costs.py`): flagging a fraud saves its whole amount, a false alarm costs only a flat EUR 5 review, "no model" means zero protection; no customer-friction cost. "Saved vs no model" is therefore an upper-bound style figure.
8. **Calibration, threshold and rule limits all use the same validation period** (the test is the independent check).
9. **Labels are instantaneous** in the evaluation (no chargeback delay), except in the merchant/category-risk features.
10. **Policy rules** add 408 reviews and no extra detected fraud on the test set (section 7.6).
11. **Weekly precision/recall in the Monitoring screenshot** falls to 0% in the last plotted week; this kit did not investigate (likely very few transactions in the partial last week: UNVERIFIED).
12. **Single machine, single seed (42).** Reproducibility is evidenced by identical metrics for the same hash (section 8.3), not by multiple seeds.

### 8.8 Other measurements (this machine; `_data/extra_facts.json`)
- Explaining one transaction with the served model, warm, mean of 20 calls: 32.3 ms; probabilities for 10,000 prepared rows: 90.6 ms.
- Model file `fraudlens/artifacts/model.joblib`: 2,433,713 bytes (900 trees of up to 23 leaves).

## 9. Testing

### 9.1 Exact test count (command: `pytest --collect-only -q`, `_data/test_inventory.txt`)
**81 tests collected.**

| Test file (under fraudlens/) | Tests |
|---|---|
| tests/test_app.py | 20 |
| tests/test_platform.py | 15 |
| tests/test_auth.py | 9 |
| tests/test_api.py | 8 |
| tests/test_rules.py | 7 |
| tests/test_features.py | 7 |
| tests/test_explain.py | 5 |
| tests/test_decision.py | 4 |
| tests/test_cases.py | 3 |
| tests/test_bootstrap.py | 2 |
| tests/test_train_repro.py | 1 |

### 9.2 Result of the full run (command: `fraudlens/.venv/Scripts/python.exe -m pytest -q`, saved in `_data/pytest_full_run.txt`)
**80 passed, 1 failed.** The failure is `tests/test_train_repro.py::test_training_is_reproducible`: `requests.exceptions.SSLError ... CERTIFICATE_VERIFY_FAILED` while contacting `api.kaggle.com` (the test calls `kagglehub`, which needs network access; a second run of that test alone failed the same way: `_data/pytest_repro_test_rerun.txt`). It is an environment (certificate/network) failure, not a model failure; an earlier run on the same day (console output only, no saved file) reported 81 passed. The test skips itself in CI because the dataset is not cached there.

### 9.3 Lint and format (commands run by `report_kit/scripts/facts_part2.py` in `fraudlens/`)
- `ruff check .` -> All checks passed!
- `ruff format --check .` -> 53 files already formatted
- Ruff rule sets selected in `fraudlens/pyproject.toml`: E, F, W, B, S (security), UP, SIM, PL; line length 120.

### 9.4 Continuous integration (`.github/workflows/fraudlens-ci.yml`)
Triggers: push or pull request touching `fraudlens/**`. Job on `ubuntu-latest`, Python 3.11: (1) checkout, (2) set up Python with pip cache, (3) `pip install -r requirements.txt -r requirements-api.txt ruff==0.17.1`, (4) `ruff check . && ruff format --check .`, (5) `pytest -q` (tests needing the Kaggle data or a trained model skip themselves).
Latest runs (command: `gh run list`, `_data/ci_runs.txt`): success on f73d82a (2026-10-31); success on ee5fcd8 (2026-10-26); success on 9b7be1a (2026-10-25); success on 173feb0 (2026-10-20).

### 9.5 Lines of code (tracked Python files only; command: `git ls-files 'fraudlens/*.py'`, counted by `report_kit/scripts/facts_common.py:loc`)
| folder | python files | physical lines | code lines (non-blank, non-comment) |
|---|---|---|---|
| fraudkit/ (package) | 18 | 3805 | 3226 |
| app/ (Streamlit) | 12 | 2969 | 2675 |
| api/ | 2 | 336 | 272 |
| tests/ | 12 | 1274 | 1010 |
| scripts/ | 5 | 1531 | 1399 |
| TOTAL (all tracked fraudlens/*.py) | 49 | 9915 | 8582 |

Not counted: Streamlit CSS inside `app/common.py` strings (counted as Python lines), YAML, Markdown, the Graphviz and report_kit files.

## 10. Timeline (from `git log`, repository `gauravv-jainn/frauddet`, branch main)

| Phase | First commit date | Last commit date | Commits | Lines added | Lines deleted |
|---|---|---|---|---|---|
| 0. Original mini project (notebook, FraudShield app, docs) | 2026-10-20 | 2026-10-20 | 1 | 1,847 | 0 |
| 1. FraudLens scaffold and data (M0-M1) | 2026-10-20 | 2026-10-20 | 2 | 582 | 0 |
| 2. Modelling and explanations (M3-M4) | 2026-10-20 | 2026-10-20 | 2 | 2,914 | 1 |
| 3. App, API and docs (M5-M7) | 2026-10-20 | 2026-10-20 | 4 | 2,231 | 301 |
| 4. Audit and enterprise hardening (M8) | 2026-10-20 | 2026-10-20 | 3 | 8,821 | 2,189 |
| 5. Policy rules and viva report (M9) | 2026-10-20 | 2026-10-20 | 1 | 3,392 | 223 |
| 6. Decision logic v2, analyst workflow, UI redesign (M10) | 2026-10-26 | 2026-10-26 | 1 | 4,665 | 460 |
| 7. Model v3: features, calibration, risk score (M11) | 2026-10-27 | 2026-10-27 | 1 | 18,128 | 1,796 |
| 8. Hosted-demo deployment preparation (M12) | 2026-10-31 | 2026-10-31 | 1 | 230 | 0 |

All commits:

| Commit | Date | Author | Subject | Files | + | - |
|---|---|---|---|---|---|---|
| d490db0 | 2026-10-20 | Gaurav Jain | Credit card fraud detection mini project: notebook, FraudShield app, docs | 9 | 1847 | 0 |
| 7ddcf5e | 2026-10-20 | gauravv-jainn | M0/M1: fraudlens scaffold, pinned env, config, Kaggle data loader | 7 | 271 | 0 |
| df3077a | 2026-10-20 | gauravv-jainn | M2: leakage-free feature pipeline + tests | 4 | 311 | 0 |
| b89f383 | 2026-10-20 | gauravv-jainn | M3: time-split leaderboard, Optuna tuning, calibration, money threshold, live rich training CLI | 9 | 2511 | 1 |
| 7e77aeb | 2026-10-20 | gauravv-jainn | M4: SHAP reason codes with real numbers, risk bands, what-if + tests | 3 | 403 | 0 |
| e665984 | 2026-10-20 | gauravv-jainn | M5: Streamlit app (6 pages), SQLite case queue, AppTest click-through | 17 | 1693 | 301 |
| 788cafb | 2026-10-20 | gauravv-jainn | M6: demo mode, FastAPI /score with API key, Dockerfile | 8 | 215 | 0 |
| d1613e0 | 2026-10-20 | gauravv-jainn | M7: README with real results, Playwright screenshots, progress log | 11 | 317 | 0 |
| e406cda | 2026-10-20 | gauravv-jainn | Root README: link to fraudlens/ | 1 | 6 | 0 |
| 3b4acd1 | 2026-10-20 | gauravv-jainn | Audit fixes + platform: auth/roles, hashed API keys, validated config, SQLite WAL + migrations | 20 | 1519 | 190 |
| fbc2f86 | 2026-10-20 | gauravv-jainn | App: login, role-gated pages, Admin/Monitoring/Account pages, case assignment; fix UI bugs | 13 | 1633 | 372 |
| c74513e | 2026-10-20 | gauravv-jainn | Drift monitoring finds non-stationary feature; atomic versioned artifacts; ops + docs | 35 | 5669 | 1627 |
| 173feb0 | 2026-10-20 | gauravv-jainn | Policy-rule guardrail for model blind spots + viva report PDF | 38 | 3392 | 223 |
| 9b7be1a | 2026-10-26 | gauravv-jainn | Decision logic v2, analyst workflow and UI redesign | 56 | 4665 | 460 |
| ee5fcd8 | 2026-10-27 | gauravv-jainn | Model v3: research-based features, smooth calibration, 0-999 risk score | 39 | 18128 | 1796 |
| f73d82a | 2026-10-31 | gauravv-jainn | Add Hugging Face Spaces deployment for the hosted demo | 10 | 230 | 0 |

*Source:* command: `git log --reverse --date=short --shortstat` (`tables/git_timeline.csv`)

Caveats: 16 commits in total; phases 0-5 (13 commits) all carry the same calendar date (2026-10-20), so commit dates are not a reliable record of how long the work took. Authors in git: `Gaurav Jain` (1 commit, the original mini project) and `gauravv-jainn` (the rest); no commit is attributed to the second team member (see open questions). The 18,128 insertions of phase 7 include generated artifacts (`artifacts/runs/*`, tables, the viva PDF) and are not hand-written code. Today's date in this environment is 2026-11-01.

## 11. Screenshots: captions and which role sees what

All 12 files are copied to `figures/screenshots/` from `fraudlens/docs/screenshots/` (taken on 2026-10-27 right after the final model was saved, by `fraudlens/scripts/screenshots.py`). Every screenshot is signed in as an **admin** account named `demo.admin`, which the script creates (`scripts/screenshots.py:66`), so the sidebar shows every page. A viewer sees fewer pages and an analyst cannot open Admin; see `tables/pages_roles.csv`. The password never appears in an image. Captions marked "no" in the last column were written from the page's code, not from inspecting the image: check them against the image before printing.

| File | Page | Roles that can open it | Caption (2-3 lines) | Image inspected? |
|---|---|---|---|---|
| figures/screenshots/0_login.png | Sign in | (no sign-in needed to see this page) | Sign-in form shown before any page. On a fresh database it becomes a first-run setup that creates the first administrator. | no: caption from fraudlens/app/common.py _login_screen |
| figures/screenshots/1_dashboard.png | Dashboard | viewer, analyst, admin | KPI tiles for the held-out test period (555,719 scored, 2,463 flagged, recall 95.9%, precision 83.5%, 7.3 false alarms per 10k) with daily alerts, decision mix (approve 553,256, review 512, block 1,951), flagged-by-category, hour/weekday heat map and top merchants. Policy rules are not included in these tiles. | yes |
| figures/screenshots/2_check_transaction.png | Check a transaction | viewer, analyst, admin (entering a transaction and what-if need analyst or admin) | The highest-risk real test transaction (EUR 856.30, misc_net, 22:57): risk score 999, Critical, action Block, with red reasons (+9.35 large amount, +7.55 amount 5.4x the 30-day average, +4.42 burst of transactions, +4.27 night) and green reasons. | yes |
| figures/screenshots/2b_what_if.png | Check a transaction: what-if | analyst, admin | What-if panel: edit amount, hour of day or merchant distance and re-score the same transaction with the card's real history. | no: caption from fraudlens/app/views/2_Check_Transaction.py |
| figures/screenshots/2c_policy_rules.png | Check a transaction: policy rules | analyst, admin | A typed transaction of EUR 12,000 at 2,500 km from home: the model alone says Low (0.01%); five policy rules raise the band to High and the action to Review (risk score 231). This is the case that motivated the rule layer. | yes |
| figures/screenshots/3_batch_scoring.png | Batch scoring | analyst, admin | Upload a CSV (or score the 500-row sample) to get a risk-ranked table with a reason per row, a CSV download and an option to send flagged rows to the case queue. | no: caption from fraudlens/app/views/3_Batch_Scoring.py and a live run in this session |
| figures/screenshots/4_case_queue.png | Case queue | viewer, analyst, admin (acting on cases needs analyst or admin; clearing the queue needs admin) | Kanban board of 24 new and 1 closed demo cases with SLA and assignee, a case detail with reasons, Confirm fraud / Mark genuine / Escalate, comments, audit trail and analyst-feedback charts (100% agreement is based on one decision). | yes |
| figures/screenshots/5_model_report.png | Model report | viewer, analyst, admin | Validation leaderboard, test PR-AUC 0.9765 with CI, reliability and cost curves, the calibration comparison, threshold vs expected-loss policy, measured effect of policy rules, global SHAP importance, data card and the limitations text. | yes |
| figures/screenshots/6_training_lab.png | Training Lab | viewer, analyst, admin (the Retrain button needs admin) | Latest training run read from artifacts: stage timeline, Optuna trial history, final leaderboard, chosen hyper-parameters, machine and library versions, stage durations. | no: caption from fraudlens/app/views/6_Training_Lab.py |
| figures/screenshots/7_monitoring.png | Monitoring | viewer, analyst, admin | Monthly PSI drift per feature against the training reference: 4 features significant and 5 moderate in 2020-12 (n_txn_7d PSI 1.030), score PSI 0.007, plus weekly precision/recall/flag rate. | yes |
| figures/screenshots/8_admin.png | Admin | admin | Admin tabs Users, API keys, Watchlists, Audit log and System; the Users tab shows add-user, role change, disable and temporary-password reset. | yes |
| figures/screenshots/9_card_360.png | Card 360 | viewer, analyst, admin (blocking a card needs analyst or admin) | One card (****5105, Hartford AL): 736 transactions, EUR 53.2k spend, 9 frauds, 8 flagged, 1 case, with a time scatter, map of locations, spend by category, usual hours and recent transactions. | yes |

## 12. Limitations, ethics, privacy, future scope

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

## 13. Viva: 25 questions with short answers

Each answer ends with the artifact or code it comes from. Numbers are filled in by `report_kit/scripts/build_tables.py` from the artifacts. Machine-readable: `tables/viva_questions.csv`.

**Q1. Why PR-AUC and not accuracy?**  
Only 0.39% of test transactions are fraud, so 'approve everything' is 99.61% accurate and catches nothing. PR-AUC looks only at how well the rare class is ranked and how many alerts are real. Test PR-AUC is 0.9765.  
*Reference:* artifacts/metrics.json split.test.fraud_rate, test.pr_auc; Saito and Rehmsmeier 2015

**Q2. Why a time-based split?**  
Fraud is detected in the future from the past. Fit = oldest 85% of the training file, validation = last 15% by time, test = the later file; the test set is used once, after calibration and the threshold are fixed.  
*Reference:* fraudlens/fraudkit/train.py:349-358; tables/dataset_splits.csv

**Q3. How do you stop data leakage in the features?**  
Every card feature is computed from strictly earlier rows of the same card (rows sorted by card, time, position). Merchant encoding is out-of-fold and fitted on the fit split; merchant/category risk ignores labels younger than 7 days. Tests truncate the future and check that features do not change.  
*Reference:* fraudkit/features.py:147-165, 92-99, 128-135, 311-320; tests/test_features.py

**Q4. What is target encoding and why out-of-fold?**  
Merchant is replaced by its smoothed fraud rate. If a row's own label were used, the feature would leak the answer, so fit rows are encoded with a mapping learned on the other 4 folds.  
*Reference:* fraudkit/features.py:284-320

**Q5. Why LightGBM?**  
On validation PR-AUC the tuned LightGBM scored 0.9872 against 0.9570 (random forest), 0.7977 (default LightGBM) and 0.7597 (logistic regression). Caveat: only LightGBM was tuned, so part of the gap is tuning, not the algorithm.  
*Reference:* tables/leaderboard_final.csv; fraudkit/train.py:397-447

**Q6. How was the model tuned?**  
Optuna TPE sampler, 30 trials, 3-fold stratified CV on PR-AUC, with 25% of genuine rows kept inside each fold for speed; best CV PR-AUC 0.9866. The final model is trained on the same 25% genuine subsample plus all frauds.  
*Reference:* fraudkit/train.py:453-477, 516-520; tables/optuna_trials_final.csv

**Q7. Why calibrate, and which method won?**  
Training on a down-sampled genuine class and weighting frauds shifts the scores away from true probabilities. Isotonic, Platt and beta were compared by out-of-sample log-loss on two time halves of validation: isotonic 0.00250, Platt 0.00159, beta 0.00162. Platt won.  
*Reference:* metrics.json calibration.comparison; fraudkit/score.py:42-85

**Q8. Why did the app first show only 0% or 100%?**  
Isotonic regression is a step function: on the validation halves it produced only 22 distinct values and 97.66% of rows sat at the floor. Platt keeps the ranking and gives 522,809 distinct values on the test set (after rounding to 12 decimals).  
*Reference:* metrics.json calibration.comparison, test.distinct_probabilities

**Q9. What is the 0-999 risk score?**  
A log-odds scale: p = 1e-6 maps to 0 and p = 0.999 to 999; about 33.4 points double the odds. The decision threshold 0.0837 is score 551. Decisions use the probability; the score is for reading.  
*Reference:* fraudkit/score.py:29-39; _data/extra_facts.json

**Q10. How is the decision threshold chosen?**  
On validation, by minimising total cost = value of missed frauds + EUR 5 per flagged transaction. That gave threshold 0.0837.  
*Reference:* metrics.json threshold; fraudkit/costs.py

**Q11. Flat threshold or expected loss, and which is served?**  
Expected-loss policy flags when p x amount >= k x review cost (k = 8). On validation the flat threshold was cheaper (8,569.66 vs 8,948.46), so it is served, even though expected loss was cheaper on test (28,475.99 vs 30,137.00); switching on test would be tuning on test. k = 8 is the largest value in the grid.  
*Reference:* tables/decision_policy_comparison.csv; fraudkit/decision.py

**Q12. What do the policy rules do and do they help?**  
They only raise the band (e.g. EUR 12,000 at 2,500 km, which the model scored Low). On the simulated test set they add 408 reviews, catch no extra fraud and raise cost by EUR 2,040; their value is protection outside the training data.  
*Reference:* tables/policy_rules_impact_test.csv; fraudkit/rules.py:87-111; screenshot 2c

**Q13. How do the reason codes work?**  
SHAP TreeExplainer gives each feature's contribution in log-odds; related features are grouped; contributions of at least 0.05 become red (top 4) or green (top 2) reasons filled with the transaction's real values. SHAP + base value reconstruct the raw model output (tested to 1e-6).  
*Reference:* fraudkit/explain.py:347-364; tests/test_explain.py; _data/shap_examples.json

**Q14. What is PSI and what did drift monitoring find?**  
PSI = sum((a - e) ln(a / e)) over bins; < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant. It showed the lifetime counter card_txn_count at PSI 4.19 (replaced by n_txn_30d) and, in 2020-12, four significant features led by n_txn_7d (PSI 1.030).  
*Reference:* fraudkit/monitoring.py:39-50; tables/drift_*.csv

**Q15. How did the model improve across runs and why?**  
Test PR-AUC 0.9682 -> 0.9722 (drifting counter replaced by n_txn_30d) -> 0.9765 (7 research-based features, Platt calibration). Between v2 and v3 the validation-chosen threshold moved 0.0238 -> 0.0837.  
*Reference:* tables/runs_comparison.csv; tables/feature_ablation.csv

**Q16. What does the 95% bootstrap CI say?**  
Test PR-AUC 0.9765, 95% CI 0.9719-0.9811 from 1000 resamples of test rows. It treats rows as independent although a card's transactions are correlated, so it is optimistic.  
*Reference:* metrics.json test; fraudkit/train.py:215-229

**Q17. What are the limits of your evaluation?**  
Simulated data; one split; 908 of the 924 test cards also appear in training (no new-card test); Optuna CV is shuffled, not time/group aware; labels arrive instantly; the money metric assumes every flagged fraud is fully recovered and 'no model' means zero protection.  
*Reference:* _data/extra_facts.json; REPORT_FACTS.md section 8

**Q18. What was wrong with the base paper's evaluation?**  
Per the team's review: a tiny balanced test set (~124 rows), preprocessing (under-sampling, correlation-based feature selection, outlier removal) before the split, and 'ROC-AUC' that matches balanced accuracy. Not verified against the paywalled paper.  
*Reference:* docs/Literature_Review_Fraud_Detection.docx section 2; INCONSISTENCIES I11

**Q19. How is the app secured?**  
PBKDF2-HMAC-SHA256 (600,000 iterations) with lockout after 5 failures for 15 minutes, 60-minute session timeout, server-side roles, hashed API keys, rate limit 120 requests per minute per key, HTML escaping and CSV-injection protection. SECURITY.md lists what is still missing for real card data.  
*Reference:* fraudkit/auth.py:35,60-72; configs/default.yaml; SECURITY.md

**Q20. Which roles exist and what can each do?**  
viewer: read-only pages; analyst: also score, work cases, feed the queue; admin: also clear the queue, retrain, manage users and API keys, view the audit log.  
*Reference:* tables/role_permission_matrix.csv; figures/role_matrix.png

**Q21. How does the API score a transaction?**  
Check key, rate limit and batch size; validate fields; build features from the card's earlier transactions; LightGBM raw score; Platt probability; risk score and band; SHAP reasons; policy rules; decision and expected loss; JSON response with the model version.  
*Reference:* figures/scoring_flow.png; api/main.py:270-327

**Q22. How is data stored?**  
SQLite in WAL mode with versioned migrations: users, api_keys, admin_audit, cases, audit, case_comments, watchlist. Card numbers are stored as the last four digits; blocked cards as a SHA-256 hash.  
*Reference:* figures/sqlite_er.png; fraudkit/db.py:14-107; fraudkit/security.py:30-40

**Q23. How was it tested?**  
81 pytest tests (80 pass; the one failure needs network access to Kaggle), ruff lint and format checks, and a GitHub Actions workflow running ruff and pytest; tests needing the dataset or a trained model skip themselves in CI.  
*Reference:* _data/pytest_full_run.txt; .github/workflows/fraudlens-ci.yml

**Q24. Is training reproducible?**  
Fixed seeds and deterministic LightGBM: three archived runs share the model-hash suffix a6506be0 with identical test metrics, and tests/test_train_repro.py compares two fast runs to 10 decimals. Wall-clock times differ (502-1,235 s) with machine load.  
*Reference:* tables/runs_comparison.csv; tests/test_train_repro.py

**Q25. What is missing before real use?**  
SSO/MFA, TLS, a server database, shared rate limiting, a secrets manager, PCI DSS scoping, model governance, trust boundary for the pickle model file, real merchant/device/IP data, delayed labels, and a fairness review (not done). The hosted demo script is prepared but was not deployed.  
*Reference:* SECURITY.md; fraudlens/deploy/space; REPORT_FACTS.md section 12

## 14. OPEN QUESTIONS FOR THE STUDENTS (only you can answer these)

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

## 15. INCONSISTENCIES found while checking numbers against artifacts and code

Rule used: a number was taken from `fraudlens/artifacts/` or from a command, never from README/PROGRESS. Where README/PROGRESS/code comments disagree, the artifact value is used and the disagreement is listed here (`tables/inconsistencies.csv`).

| ID | Where | Claim or value | Evidence | Severity |
|---|---|---|---|---|
| I1 | PROGRESS.md M11 | Risk score: 'genuine median 25' | Recomputed from artifacts/test_scored.parquet with FraudModel.risk_score: genuine median 24 (_data/extra_facts.json) | minor |
| I2 | fraudkit/explain.py:73 | Comment says the 0-999 risk score is a 'rank against the validation period' | fraudkit/score.py:29-39: the score is a log-odds scale (p=1e-6 -> 0, p=0.999 -> 999), not a rank (the rank version was replaced, PROGRESS M11) | minor (stale comment) |
| I3 | app/views/5_Model_Report.py:309-312 (visible in screenshot 5) | Limitations text says 'Drift is not monitored' and that 'Isotonic calibration and the threshold are both fitted on' validation | app/views/7_Monitoring.py implements monthly PSI drift; metrics.json calibration.method = platt. Alerting/retraining automation really is absent, but the text is stale | medium (the app contradicts itself) |
| I4 | fraudkit/features.py:65 | Comment: n_txn_30d 'is stationary' | tables/drift_test_period_by_month.csv: n_txn_30d is the top-PSI feature in 2020-07/08/10/11 (0.11-0.16, moderate) and PSI 0.517 in 2020-12 (significant); whole-test-period PSI is 0.0878 (stable) | minor |
| I5 | PROGRESS.md M3/M8/M9/M10 and README history | Numbers quoted for earlier models (e.g. M10: k = 3, validation cost 8,838 vs 9,338; M9: 2,548 -> 3,159 flags; M8: threshold 0.0238) | The final artifacts differ because the model changed: k = 8, validation cost 8,569.66 vs 8,948.46, policy rules 2,463 -> 2,871 flags (tables/policy_rules_impact_test.csv). Use only the final-artifact numbers in the report | medium (superseded numbers) |
| I6 | PROGRESS.md M3 | 'Full run ... 612 s' | Archived first run 20261020-153708-3c73d465 has total_seconds 562.7 | minor |
| I7 | Final run 20261026-184402-d37e9778 | total_seconds 68,988.9 | Archived run 20261025-233115-3abe0ebd has the same test metrics (PR-AUC 0.9764813286) in 1,236.2 s; the sum of Optuna trial seconds in the final run is 68,879 s, so the long time is real elapsed time (the README says the computer was asleep). The two model files have different hash suffixes (d37e9778 vs 3abe0ebd) although metrics agree to ~1e-15; the cause was NOT investigated | minor (use 1,236 s for training time) |
| I8 | artifacts/metrics.json test.at_policy.threshold and decision.*.test.threshold | Value 0.5 | These are placeholders: fraudkit/decision.py:64 and train code call confusion_at(y, flags, amt, 0.5, review_cost) with 0/1 flags. The real threshold is 0.0837 (metrics.json threshold.value) | minor (misleading field) |
| I9 | README 'distinct probabilities' | 522,809 distinct probabilities on the test set | fraudkit/train.py:710 counts distinct values AFTER rounding to 12 decimals; artifacts/test_scored.parquet has 555,719 distinct raw values (one per row) | minor (definition) |
| I10 | README / PROGRESS test count | '79 tests passing' (PROGRESS M11) and 81 earlier today | pytest today: 81 collected = 80 passed + 1 failed (tests/test_train_repro.py, SSLError reaching api.kaggle.com). The test needs network access; it is skipped in CI because the dataset is not cached there | medium (report the failure honestly) |
| I11 | docs/Literature_Review_Fraud_Detection.docx reference [4] | Authors listed as '... G. V. P. Reddy, C. V. N. M. Praneeth, A. R. Kumar'; four classifiers incl. Decision Tree | Crossref/Semantic Scholar list the order D. Tanouz, R. Raja Subramanian, D. Eswar, G. V. Parameswara Reddy, A. Ranjith Kumar, CH. V. N. M. Praneeth; the abstract names only logistic regression, random forest and Naive Bayes | medium (fix the reference before submission) |
| I12 | fraudlens/scripts/build_report.py section 15 (viva report) and PROGRESS M11 | arXiv 1909.01185 supports 'amount and elapsed time ... log z-score'; arXiv 2503.20477 supports 'circular time' | The abstracts (arXiv API) say: 1909.01185 = HMM features from sequences of amounts / elapsed time; 2503.20477 = moving-window confidence intervals and fraud-risk scoring. Neither abstract mentions a log z-score or circular time encoding; full texts were not read -> UNVERIFIED | medium |
| I13 | Root README.md FraudLens paragraph | 'a six-page Streamlit app' | fraudlens/app/common.py PAGES lists 10 pages (tables/pages_roles.csv) | minor (root README not edited, by instruction) |
| I14 | docs/screenshots/*.png | Signed-in user 'demo.admin' | The screenshots were taken with an auto-created admin account (fraudlens/scripts/screenshots.py:66), so every page looks admin-only-capable; viewer/analyst views differ (tables/pages_roles.csv) | info |

Checks that agreed (no inconsistency): test PR-AUC, CI, ROC-AUC, precision, recall, flagged, cost and savings in the README Results block equal `metrics.json` (also recomputed independently from `test_scored.parquet` in section 17); the PROGRESS M3 validation leaderboard equals the first archived run; the README's "card_txn_count PSI 3.7-6.1 per test month" equals `tables/drift_card_txn_count_by_month.csv` (3.71-6.11).

## 16. Template cross-reference and inputs for the sustainability / SDG / budget sections

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
- **Compute:** training used CPU only (`n_jobs` = CPU count in `fraudkit/train.py:261`; no GPU in `metrics.json` machine info), 32 threads; archived training run 1,236 s; the model file is 2.43 MB and one explained transaction takes about 32 ms on this machine. **Energy use (kWh) and emissions: NOT MEASURED.**
- **Social:** 7.3 false alarms per 10,000 genuine transactions (406 genuine test transactions flagged); explanations and an audit trail for every case; privacy measures in 12.3; fairness not assessed (12.2).
- **Economic (model-side, simulated data):** missed fraud EUR 17,822 and review cost EUR 12,315 against EUR 1,133,325 of fraud in the test period (section 8.1); assumptions in 8.7 item 7.
- **SDG candidates:** the UN page https://sdgs.un.org/goals/goal16 states target 16.4: "By 2030, significantly reduce illicit financial and arms flows, strengthen the recovery and return of stolen assets and combat all forms of organized crime" (text fetched and matched). Whether card-fraud detection supports 16.4 directly is for you to argue; the project does not measure that. Wording of targets for any other SDG (for example 8 or 9) was NOT verified here.
- **Trade-offs already visible:** more review (higher recall) means more customer friction and analyst cost (threshold and expected-loss tables in section 7.5).

### 16.2 Inputs for the budget table (template 8.1; amounts in rupees are NOT FOUND)
| Item | Fact | Source |
|---|---|---|
| Dataset | free public Kaggle dataset | `fraudkit/data.py` |
| Software | open-source Python libraries (section 1.4) | `requirements*.txt` |
| Hosting | none running; a free Hugging Face Space is prepared (PROGRESS M12 says the free tier offers 16 GB RAM; not re-verified in this kit) | `deploy/space`, `PROGRESS.md` M12 |
| Hardware / electricity | the students' own computer; cost NOT FOUND | - |

## 17. Self-check log (run by `report_kit/scripts/build_facts.py`)

| Check | Expected | Observed | Result |
|---|---|---|---|
| Test PR-AUC recomputed from test_scored.parquet (sklearn) | 0.976481 (metrics.json) | 0.976481 | PASS |
| Test ROC-AUC recomputed | 0.999468 | 0.999468 | PASS |
| Test Brier (calibrated) recomputed | 0.00036797 | 0.00036797 | PASS |
| Flagged / TP / FP / FN at threshold recomputed | 2463 / 2057 / 406 / 88 | 2463 / 2057 / 406 / 88 | PASS |
| Caught fraud amount recomputed | 1115502.68 | 1115502.68 | PASS |
| Test rows and frauds | 555719 / 2145 | 555719 / 2145 | PASS |
| Test count in text equals collected tests | 81 | 81 | PASS |
| Screenshots copied | 12 | 12 | PASS |
| Generated diagrams exist (5) | 5 | 5 | PASS |
| Abstract length <= 200 words | <= 200 | 190 | PASS |
| Narrative: 406 flagged genuine, 7.3 false alarms per 10k | 406 / 7.3 | 406 / 7.3 | PASS |
| Narrative: rules add 408 reviews | 408 | 408 | PASS |
| Narrative: age rank 9, gender rank 25 of 28 | 9 / 25 / 28 | 9 / 25 / 28 | PASS |
| Narrative: threshold moved 0.0238 -> 0.0837 | 0.0238 / 0.0837 | 0.0238 / 0.0837 | PASS |
| Narrative: 16 commits, 15 by gauravv-jainn, 13 dated 2026-10-20 | 16 / 15 / 13 | 16 / 15 / 13 | PASS |
| Narrative: 7 feature tests, 8 API tests, 10 pages | 7 / 8 / 10 | 7 / 8 / 10 | PASS |
| Every reference with a DOI in the kit is registry-verified | all VERIFIED rows have a DOI | 17 verified, 0 not | PASS |
| Every file path in backticks exists | 0 missing | 0 missing | PASS |

Tables are generated by code from `fraudlens/artifacts/` and `report_kit/tables/`. The "Narrative" rows above re-check the numbers that were typed by hand into prose. Numbers copied from documents (base-paper table, PROGRESS.md) are labelled as such where they appear.
