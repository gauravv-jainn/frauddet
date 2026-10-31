# Figure and table index

Suggested numbering for the report. Sources of every number are named in `REPORT_FACTS.md`.

## Figures

| No. | File | What it shows | Source |
|---|---|---|---|
| Figure 1 | `figures/architecture.png` | System architecture: users, interfaces, fraudkit modules, storage, training and deployment | `figures/diagrams_src/architecture.dot` |
| Figure 2 | `figures/scoring_flow.png` | Life of one POST /v1/score request | `figures/diagrams_src/scoring_flow.dot` |
| Figure 3 | `figures/sqlite_er.png` | SQLite entity-relationship diagram (7 tables) | `figures/diagrams_src/sqlite_er.dot` |
| Figure 4 | `figures/training_pipeline_time_split.png` | Time-based split with row counts and the 9 training stages with measured seconds | `figures/diagrams_src/make_matplotlib_figures.py` |
| Figure 5 | `figures/role_matrix.png` | Role / permission matrix | `figures/diagrams_src/make_matplotlib_figures.py` |
| Figure 6 | `figures/screenshots/0_login.png` | Screenshot: Sign in | `fraudlens/docs/screenshots/` |
| Figure 7 | `figures/screenshots/1_dashboard.png` | Screenshot: Dashboard | `fraudlens/docs/screenshots/` |
| Figure 8 | `figures/screenshots/2_check_transaction.png` | Screenshot: Check a transaction | `fraudlens/docs/screenshots/` |
| Figure 9 | `figures/screenshots/2b_what_if.png` | Screenshot: Check a transaction: what-if | `fraudlens/docs/screenshots/` |
| Figure 10 | `figures/screenshots/2c_policy_rules.png` | Screenshot: Check a transaction: policy rules | `fraudlens/docs/screenshots/` |
| Figure 11 | `figures/screenshots/3_batch_scoring.png` | Screenshot: Batch scoring | `fraudlens/docs/screenshots/` |
| Figure 12 | `figures/screenshots/4_case_queue.png` | Screenshot: Case queue | `fraudlens/docs/screenshots/` |
| Figure 13 | `figures/screenshots/5_model_report.png` | Screenshot: Model report | `fraudlens/docs/screenshots/` |
| Figure 14 | `figures/screenshots/6_training_lab.png` | Screenshot: Training Lab | `fraudlens/docs/screenshots/` |
| Figure 15 | `figures/screenshots/7_monitoring.png` | Screenshot: Monitoring | `fraudlens/docs/screenshots/` |
| Figure 16 | `figures/screenshots/8_admin.png` | Screenshot: Admin | `fraudlens/docs/screenshots/` |
| Figure 17 | `figures/screenshots/9_card_360.png` | Screenshot: Card 360 | `fraudlens/docs/screenshots/` |

## Tables (CSV in `tables/`)

| No. | File | Rows | What it contains |
|---|---|---|---|
| Table 1 | `tables/api_endpoints.csv` | 6 | API endpoints |
| Table 2 | `tables/base_paper_vs_ours.csv` | 5 | Base-paper numbers vs reproduced vs FraudLens |
| Table 3 | `tables/calibration_comparison_final.csv` | 3 | Isotonic vs Platt vs beta (out-of-sample) |
| Table 4 | `tables/dataset_columns_raw.csv` | 23 | Raw Kaggle columns: dtype, nulls, distinct |
| Table 5 | `tables/dataset_splits.csv` | 4 | Fit / validation / test / raw train sizes |
| Table 6 | `tables/decision_expected_loss_k_grid.csv` | 10 | Expected-loss policy: k grid on validation |
| Table 7 | `tables/decision_policy_comparison.csv` | 4 | Threshold vs expected loss on validation and test |
| Table 8 | `tables/drift_2020_12_all_features.csv` | 29 | PSI of all features in December 2020 |
| Table 9 | `tables/drift_card_txn_count_by_month.csv` | 7 | PSI of the retired feature card_txn_count per month |
| Table 10 | `tables/drift_test_period_by_month.csv` | 7 | Top PSI feature and alert counts per month |
| Table 11 | `tables/drift_test_period_overall.csv` | 29 | PSI per feature over the whole test period |
| Table 12 | `tables/feature_ablation.csv` | 13 | Validation ablation of candidate features |
| Table 13 | `tables/features_catalog.csv` | 28 | 28 model features: definition, code line, importance |
| Table 14 | `tables/fraud_rate_by_category_train.csv` | 14 | Fraud rate per category (training file) |
| Table 15 | `tables/git_timeline.csv` | 16 | All commits with line stats |
| Table 16 | `tables/global_importance_final.csv` | 28 | Mean |SHAP| of all 28 features |
| Table 17 | `tables/inconsistencies.csv` | 14 | Disagreements between docs, comments and artifacts |
| Table 18 | `tables/leaderboard_by_run.csv` | 24 | Validation PR-AUC/ROC-AUC/Brier of every model in every run |
| Table 19 | `tables/leaderboard_final.csv` | 4 | Final validation leaderboard |
| Table 20 | `tables/literature_survey.csv` | 19 | Literature survey table |
| Table 21 | `tables/optuna_trials_final.csv` | 30 | 30 Optuna trials of the final run |
| Table 22 | `tables/pages_roles.csv` | 10 | App pages and which roles can open them |
| Table 23 | `tables/policy_rules_impact_test.csv` | 2 | Model policy alone vs with rules (test) |
| Table 24 | `tables/policy_rules_per_rule_test.csv` | 6 | Hits and extra reviews per rule (test) |
| Table 25 | `tables/references_verification.csv` | 28 | First verification pass (search-based) output |
| Table 26 | `tables/references_verified_final.csv` | 17 | Crossref-verified bibliographic records |
| Table 27 | `tables/reliability_test_final.csv` | 24 | Test reliability table before/after calibration |
| Table 28 | `tables/role_permission_matrix.csv` | 9 | Permission x role |
| Table 29 | `tables/rules_spend_spike_grid.csv` | 20 | Grid used to pick the spending-spike rule limits |
| Table 30 | `tables/runs_comparison.csv` | 6 | Every archived training run: quality, money, time |
| Table 31 | `tables/screenshots_captions.csv` | 12 | Screenshot captions and roles |
| Table 32 | `tables/tech_stack.csv` | 31 | Tool and library versions |
| Table 33 | `tables/viva_questions.csv` | 25 | 25 viva questions and answers |

## Code excerpts

`code_snippets.md` has 15 verbatim excerpts S1-S15 (each at most 25 lines, with path and line range).
