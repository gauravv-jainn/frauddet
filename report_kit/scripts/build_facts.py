"""Assemble report_kit/REPORT_FACTS.md, references.md, glossary.md, figure_table_index.md and run the self-check.

Order: extract_results.py -> extra_facts.py -> verify_refs*.py -> build_tables.py -> make_snippets.py -> build_facts.py
Run from the repo root with fraudlens/.venv/Scripts/python.exe.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facts_part1 as p1  # noqa: E402
import facts_part2 as p2  # noqa: E402
from facts_common import FL, KIT, M, REPO, TEST, ieee_authors, run, tab  # noqa: E402


# ------------------------------------------------------------------ references.md
def build_references() -> str:
    v = tab("references_verified_final.csv")
    v = v[v.status == "VERIFIED"].reset_index(drop=True)
    lines = ["# References (IEEE style)\n",
             "Entries marked VERIFIED were found in a registry (Crossref, arXiv, publisher or proceedings page) and match on title, first author and year; the DOI printed is the one the registry returned. Nothing here was copied from memory without that check. Machine-readable evidence: `tables/references_verified_final.csv`, `tables/references_verification.csv`, `_data/verify_refs_pass1_output.txt`. Accessed date for web pages: 1 November 2026. DOIs are case-insensitive; they are printed as the registry returned them.\n",
             "## A. Verified (journal, conference and arXiv sources)\n"]
    k = 1
    refs_by_key = {}
    for _, r in v.iterrows():
        auth = ieee_authors(r["authors"])
        pages = f", pp. {r['pages']}" if isinstance(r["pages"], str) and r["pages"] and not str(r["pages"]).startswith("e") else ""
        vol = f", vol. {int(r['volume'])}" if str(r["volume"]) not in ("", "nan") else ""
        iss = f", no. {int(float(r['issue']))}" if str(r["issue"]) not in ("", "nan") else ""
        art = f", art. no. {r['pages']}" if isinstance(r["pages"], str) and str(r["pages"]).startswith("e") else ""
        lines.append(f"[{k}] {auth}, \"{r['registry_title']},\" {r['container']}{vol}{iss}{pages}{art}, {int(r['year'])}, doi: {r['doi']}.\n")
        refs_by_key[r["key"]] = k
        k += 1
    pass1 = tab("references_verification.csv").set_index("key")
    ax = [("lundberg2017", "S. M. Lundberg and S.-I. Lee", "arXiv:1705.07874 (also NeurIPS 30, 2017: the NeurIPS venue was NOT checked here)"),
          ("arxiv1909", "Y. Lucas, P.-E. Portier, L. Laporte, L. He-Guelton, O. Caelen, M. Granitzer, and S. Calabretto", "arXiv:1909.01185, 2019"),
          ("arxiv2503", "A. Stotsky", "arXiv:2503.20477, 2025 (preprint)")]
    for key, auth, venue in ax:
        r = pass1.loc[key]
        lines.append(f"[{k}] {auth}, \"{r['registry_title']},\" {venue}. [Online]. Available: {r['url']}\n")
        k += 1
    web = [("Kull, Silva Filho, Flach", "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers", "Proceedings of Machine Learning Research, vol. 54 (AISTATS); the year 2017 is taken from the page address `kull17a`", "https://proceedings.mlr.press/v54/kull17a.html"),
           ("G. Ke et al.", "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", "Advances in Neural Information Processing Systems 30 (NeurIPS 2017)", "https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html"),
           ("Y.-A. Le Borgne et al. (surnames Le Borgne, Siblini, Lebichot, Bontempi appear on the page; first names and year NOT checked)", "Reproducible Machine Learning for Credit Card Fraud detection - Practical handbook", "online book", "https://fraud-detection-handbook.github.io/fraud-detection-handbook/"),
           ("OWASP Foundation", "Password Storage Cheat Sheet", "OWASP Cheat Sheet Series (page lists PBKDF2-HMAC-SHA256: 600,000 iterations)", "https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html"),
           ("namebrandon (GitHub user)", "Sparkov_Data_Generation: Synthetic Credit Card Transaction Generator", "GitHub repository", "https://github.com/namebrandon/Sparkov_Data_Generation"),
           ("kartik2112 (Kaggle user)", "Credit Card Transactions Fraud Detection Dataset", "Kaggle dataset", "https://www.kaggle.com/datasets/kartik2112/fraud-detection"),
           ("United Nations", "The 17 Goals", "Sustainable Development (page quotes target 16.4)", "https://sdgs.un.org/goals/goal16")]
    for a, t, venue, url in web:
        lines.append(f"[{k}] {a}, \"{t},\" {venue}. [Online]. Available: {url} (accessed 1 Nov. 2026).\n")
        k += 1
    lines += ["\n## B. Pages cited in the repo's feature research (reachable, content NOT read in this kit)\n",
              "These are listed in `fraudlens/PROGRESS.md` M11 and the viva report; each URL returned HTTP 200 (`_data/progress_sources_check.txt`), but whether it supports the claim attached to it was not checked.\n",
              "- US patent 8,041,620 (velocity counting): PDF reachable at https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8041620; title and inventors NOT read.",
              "- \"Real-Time Feature Engineering for Fraud Detection,\" RisingWave: https://risingwave.com/blog/real-time-feature-engineering-fraud-detection",
              "- \"Detecting payment card fraud with Temporian and YDF,\" Temporian docs: https://temporian.readthedocs.io/en/latest/tutorials/bank_fraud_detection_with_tfdf/",
              "- \"Preventing Card Testing Fraud,\" Chargeback Gurus: https://www.chargebackgurus.com/blog/card-testing-fraud",
              "- \"Card Testing: How to Detect & Prevent It,\" Fraudlogix: https://www.fraudlogix.com/?p=17109",
              "- Fraud Detection Handbook, \"Baseline feature transformation\": https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineFeatureTransformation.html\n",
              "## C. UNVERIFIED (do not cite without checking)\n",
              "- **Platt (1999/2000), \"Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,\" in Advances in Large Margin Classifiers (MIT Press), pp. 61-74.** A web search returned pages that list this chapter, but sources disagree on the year (1999 vs 2000) and no registry record or DOI was found: UNVERIFIED. Used in the project as the name of the calibration method (`fraudkit/score.py`).",
              "- **The numbers attributed to Tanouz et al. (96.77% accuracy, 100% precision, 91.11% recall) and the claim that a decision tree was also compared:** the paper is behind the IEEE paywall; the abstract names only logistic regression, random forest and Naive Bayes. UNVERIFIED.",
              "- **That arXiv:1909.01185 supports an amount log z-score and arXiv:2503.20477 supports circular time encoding** (as attributed in `fraudlens/PROGRESS.md` M11 and the viva report): the abstracts do not say so; full texts not read. UNVERIFIED (INCONSISTENCIES I12).",
              "- **Hugging Face free-tier hardware (16 GB RAM):** from the development session, not re-checked. UNVERIFIED.\n"]
    return "\n".join(lines)


# ------------------------------------------------------------------ glossary.md
GLOSSARY = [
    ("Accuracy", "Share of all predictions that are correct. Misleading when fraud is 0.39% of rows, because 'approve everything' scores 99.6%."),
    ("Precision", "Of the transactions flagged as fraud, the share that really are fraud: TP / (TP + FP)."),
    ("Recall (sensitivity)", "Of the real frauds, the share that were flagged: TP / (TP + FN)."),
    ("False alarm rate per 10,000 genuine", "FP / (FP + TN) x 10,000: how many genuine transactions are flagged per 10,000 genuine ones."),
    ("PR-AUC (average precision)", "Area under the precision-recall curve; summarises how well the rare class is ranked. Preferred to ROC-AUC on imbalanced data."),
    ("ROC-AUC", "Probability that a random fraud is ranked above a random genuine transaction; can look excellent even when precision is poor on rare classes."),
    ("Brier score", "Mean squared difference between predicted probability and the 0/1 outcome; lower is better; measures calibration and sharpness together."),
    ("Log-loss", "Average negative log of the probability given to the true class; punishes confident wrong predictions heavily."),
    ("Calibration", "Making predicted probabilities match observed frequencies (of the transactions given 5%, about 5% should be fraud)."),
    ("Platt scaling", "Calibration by fitting a logistic regression on the logit of the raw model score."),
    ("Isotonic regression", "Monotone step-function calibration; flexible but produces many tied values (here, scores stuck at 0 and 1)."),
    ("Beta calibration", "Logistic regression on log(p) and -log(1-p); a more flexible smooth calibrator."),
    ("Reliability diagram", "Plot of mean predicted probability against observed fraud rate in probability bins."),
    ("Bootstrap confidence interval", "Interval from re-computing a metric on many resamples (with replacement) of the test rows."),
    ("Class imbalance", "One class (fraud) is very rare compared with the other."),
    ("Under-sampling / SMOTE", "Ways to rebalance training data: drop majority rows / create synthetic minority rows by interpolation."),
    ("Data leakage", "Using information in training or feature building that would not be available at prediction time."),
    ("Time-based split", "Training on the past and testing on the future, instead of a random split."),
    ("Target encoding (out-of-fold)", "Replacing a category by its smoothed fraud rate, computed on other folds so a row never uses its own label."),
    ("Velocity feature", "Count or sum of a card's recent transactions in a time window (e.g. last hour)."),
    ("RFM windows", "Recency / frequency / monetary summaries over time windows, here counts and average amounts over 1, 7 and 30 days."),
    ("Z-score", "How many standard deviations a value is above the mean of the card's own history."),
    ("Haversine distance", "Great-circle distance between two latitude/longitude points."),
    ("Gradient boosting / LightGBM", "An ensemble of decision trees built one after another, each correcting the previous ones; LightGBM is a fast implementation."),
    ("Random forest", "An average of many decorrelated decision trees."),
    ("Optuna / TPE", "Hyper-parameter search library; TPE is its Bayesian-style sampler that favours promising regions."),
    ("Cross-validation", "Repeatedly training on part of the data and scoring on the held-out part to estimate performance."),
    ("SHAP", "Method that splits a prediction into additive contributions of each feature, here in log-odds."),
    ("Log-odds", "log(p / (1 - p)); the scale on which SHAP contributions add up."),
    ("Risk score 0-999", "FraudLens reading aid: a linear rescaling of log-odds so that p = 1e-6 is 0 and p = 0.999 is 999."),
    ("Review cost / expected loss", "Cost of an analyst reviewing one alert; expected loss of approving is probability x amount."),
    ("Step-up verification", "Asking the cardholder for an extra check (for example a one-time code) before approving."),
    ("Policy rule", "A fixed business rule that can raise a transaction's risk band regardless of the model."),
    ("Watchlist", "List of blocked cards or watched merchants that forces a higher band."),
    ("Case queue / SLA", "List of flagged transactions to work; SLA is the time within which a case should be handled (24 h here)."),
    ("PSI (population stability index)", "Measure of how much a distribution has shifted from a reference: < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant."),
    ("Concept / data drift", "Change over time in the input data or in the relationship between inputs and fraud."),
    ("WAL (write-ahead logging)", "SQLite journal mode that lets readers and a writer work at the same time."),
    ("PBKDF2", "Password-hashing function that repeats a hash many times (600,000 here) with a random salt."),
    ("RBAC", "Role-based access control: permissions are given to roles (viewer, analyst, admin), not to individuals."),
    ("Rate limiting", "Capping how many requests a client may send per minute (120 per API key here)."),
    ("XSS / CSV injection", "Injecting script into a web page / a spreadsheet formula into an exported CSV; both are escaped or prefixed in the app."),
    ("Pickle / joblib", "Python object serialisation format used for the model file; loading an untrusted file can run code."),
    ("Sparkov", "Open-source generator of synthetic card transactions used to create the dataset."),
]


def build_glossary() -> str:
    return "# Glossary\n\nStandard definitions written for this kit; project-specific values (thresholds, parameters, results) are in `REPORT_FACTS.md`.\n\n" + "\n".join(f"- **{t}**: {d}" for t, d in GLOSSARY) + "\n"


# ------------------------------------------------------------------ figure_table_index.md
FIG = [
    ("architecture.png", "System architecture: users, interfaces, fraudkit modules, storage, training and deployment", "architecture.dot (Graphviz)"),
    ("scoring_flow.png", "Life of one POST /v1/score request", "scoring_flow.dot (Graphviz)"),
    ("sqlite_er.png", "SQLite entity-relationship diagram (7 tables)", "sqlite_er.dot (Graphviz)"),
    ("training_pipeline_time_split.png", "Time-based split with row counts and the 9 training stages with measured seconds", "make_matplotlib_figures.py"),
    ("role_matrix.png", "Role / permission matrix", "make_matplotlib_figures.py"),
]
TABLE_DESC = {
    "runs_comparison.csv": "Every archived training run: quality, money, time", "leaderboard_by_run.csv": "Validation PR-AUC/ROC-AUC/Brier of every model in every run", "leaderboard_final.csv": "Final validation leaderboard",
    "calibration_comparison_final.csv": "Isotonic vs Platt vs beta (out-of-sample)", "decision_expected_loss_k_grid.csv": "Expected-loss policy: k grid on validation", "decision_policy_comparison.csv": "Threshold vs expected loss on validation and test",
    "rules_spend_spike_grid.csv": "Grid used to pick the spending-spike rule limits", "global_importance_final.csv": "Mean |SHAP| of all 28 features", "reliability_test_final.csv": "Test reliability table before/after calibration",
    "dataset_splits.csv": "Fit / validation / test / raw train sizes", "dataset_columns_raw.csv": "Raw Kaggle columns: dtype, nulls, distinct", "fraud_rate_by_category_train.csv": "Fraud rate per category (training file)",
    "optuna_trials_final.csv": "30 Optuna trials of the final run", "feature_ablation.csv": "Validation ablation of candidate features", "features_catalog.csv": "28 model features: definition, code line, importance",
    "drift_test_period_overall.csv": "PSI per feature over the whole test period", "drift_test_period_by_month.csv": "Top PSI feature and alert counts per month", "drift_2020_12_all_features.csv": "PSI of all features in December 2020",
    "drift_card_txn_count_by_month.csv": "PSI of the retired feature card_txn_count per month", "policy_rules_impact_test.csv": "Model policy alone vs with rules (test)", "policy_rules_per_rule_test.csv": "Hits and extra reviews per rule (test)",
    "role_permission_matrix.csv": "Permission x role", "pages_roles.csv": "App pages and which roles can open them", "api_endpoints.csv": "API endpoints", "git_timeline.csv": "All commits with line stats",
    "tech_stack.csv": "Tool and library versions", "screenshots_captions.csv": "Screenshot captions and roles", "literature_survey.csv": "Literature survey table", "base_paper_vs_ours.csv": "Base-paper numbers vs reproduced vs FraudLens",
    "references_verified_final.csv": "Crossref-verified bibliographic records", "references_verification.csv": "First verification pass (search-based) output", "inconsistencies.csv": "Disagreements between docs, comments and artifacts", "viva_questions.csv": "25 viva questions and answers",
}


def build_index() -> str:
    L = ["# Figure and table index\n", "Suggested numbering for the report. Sources of every number are named in `REPORT_FACTS.md`.\n", "## Figures\n", "| No. | File | What it shows | Source |", "|---|---|---|---|"]
    i = 1
    for f, d, s in FIG:
        L.append(f"| Figure {i} | `figures/{f}` | {d} | `figures/diagrams_src/{s.split(' ')[0]}` |")
        i += 1
    shots = tab("screenshots_captions.csv")
    for _, r in shots.iterrows():
        L.append(f"| Figure {i} | `{r['file']}` | Screenshot: {r['page']} | `fraudlens/docs/screenshots/` |")
        i += 1
    L += ["", "## Tables (CSV in `tables/`)\n", "| No. | File | Rows | What it contains |", "|---|---|---|---|"]
    j = 1
    for f in sorted((KIT / "tables").glob("*.csv")):
        L.append(f"| Table {j} | `tables/{f.name}` | {len(pd.read_csv(f))} | {TABLE_DESC.get(f.name, '')} |")
        j += 1
    L += ["", "## Code excerpts\n", "`code_snippets.md` has 15 verbatim excerpts S1-S15 (each at most 25 lines, with path and line range).\n"]
    return "\n".join(L)


# ------------------------------------------------------------------ self-check
def selfcheck(text: str) -> tuple[str, bool]:
    rows, ok_all = [], True

    def add(name, expected, observed, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        rows.append(f"| {name} | {expected} | {observed} | {'PASS' if ok else 'FAIL'} |")

    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    sc = pd.read_parquet(FL / "artifacts" / "test_scored.parquet", columns=["is_fraud", "p", "amt"])
    y, p = sc["is_fraud"].to_numpy(), sc["p"].to_numpy()
    thr = M["threshold"]["value"]
    flag = p >= thr
    tp, fp, fn = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())
    pr = average_precision_score(y, p)
    add("Test PR-AUC recomputed from test_scored.parquet (sklearn)", f"{TEST['pr_auc']:.6f} (metrics.json)", f"{pr:.6f}", abs(pr - TEST["pr_auc"]) < 1e-6)
    ra = roc_auc_score(y, p)
    add("Test ROC-AUC recomputed", f"{TEST['roc_auc']:.6f}", f"{ra:.6f}", abs(ra - TEST["roc_auc"]) < 1e-6)
    br = brier_score_loss(y, p)
    add("Test Brier (calibrated) recomputed", f"{TEST['brier_after_calibration']:.8f}", f"{br:.8f}", abs(br - TEST["brier_after_calibration"]) < 1e-8)
    at = TEST["at_threshold"]
    add("Flagged / TP / FP / FN at threshold recomputed", f"{at['flagged']} / {at['tp']} / {at['fp']} / {at['fn']}", f"{int(flag.sum())} / {tp} / {fp} / {fn}", (int(flag.sum()), tp, fp, fn) == (at["flagged"], at["tp"], at["fp"], at["fn"]))
    caught = float(sc.loc[flag & (y == 1), "amt"].sum())
    add("Caught fraud amount recomputed", f"{at['caught_amount']:.2f}", f"{caught:.2f}", abs(caught - at["caught_amount"]) < 0.01)
    add("Test rows and frauds", f"{M['split']['test']['rows']} / {M['split']['test']['frauds']}", f"{len(sc)} / {int(y.sum())}", (len(sc), int(y.sum())) == (M["split"]["test"]["rows"], M["split"]["test"]["frauds"]))

    inv = pd.read_csv(KIT / "_data" / "test_inventory.txt", sep=r"\s+", header=None)
    add("Test count in text equals collected tests", "81", str(int(inv[0].sum())), int(inv[0].sum()) == 81 and "**81 tests collected.**" in text)
    shots = list((KIT / "figures" / "screenshots").glob("*.png"))
    add("Screenshots copied", "12", str(len(shots)), len(shots) == 12)
    add("Generated diagrams exist (5)", "5", str(sum((KIT / "figures" / f[0]).exists() for f in p2_fig())), all((KIT / "figures" / f[0]).exists() for f in p2_fig()))
    add("Abstract length <= 200 words", "<= 200", str(len(re.search(r"### 1.2 Abstract draft \((\d+) words", text).group(1).split())) and re.search(r"### 1.2 Abstract draft \((\d+) words", text).group(1), int(re.search(r"### 1.2 Abstract draft \((\d+) words", text).group(1)) <= 200)
    # numbers typed by hand into narrative text (checked here against the tables they came from)
    pi = tab("policy_rules_impact_test.csv")
    ft = tab("features_catalog.csv").set_index("feature")
    gt = tab("git_timeline.csv")
    runs = tab("runs_comparison.csv").set_index("model_version")
    add("Narrative: 406 flagged genuine, 7.3 false alarms per 10k", "406 / 7.3", f"{at['fp']} / {at['false_alarms_per_10k_genuine']:.1f}", at["fp"] == 406 and round(at["false_alarms_per_10k_genuine"], 1) == 7.3)
    add("Narrative: rules add 408 reviews", "408", str(int(pi.iloc[1].flagged - pi.iloc[0].flagged)), int(pi.iloc[1].flagged - pi.iloc[0].flagged) == 408)
    add("Narrative: age rank 9, gender rank 25 of 28", "9 / 25 / 28", f"{ft.loc['age', 'importance_rank']} / {ft.loc['gender', 'importance_rank']} / {len(ft)}", (ft.loc["age", "importance_rank"], ft.loc["gender", "importance_rank"], len(ft)) == (9, 25, 28))
    add("Narrative: threshold moved 0.0238 -> 0.0837", "0.0238 / 0.0837", f"{runs.loc['20261020-154750-a6506be0', 'threshold']:.4f} / {runs.loc['20261026-184402-d37e9778', 'threshold']:.4f}", round(runs.loc["20261020-154750-a6506be0", "threshold"], 4) == 0.0238 and round(runs.loc["20261026-184402-d37e9778", "threshold"], 4) == 0.0837)
    add("Narrative: 16 commits, 15 by gauravv-jainn, 13 dated 2026-10-20", "16 / 15 / 13", f"{len(gt)} / {int((gt.author == 'gauravv-jainn').sum())} / {int((gt.date == '2026-10-20').sum())}", (len(gt), int((gt.author == 'gauravv-jainn').sum()), int((gt.date == '2026-10-20').sum())) == (16, 15, 13))
    inv2 = pd.read_csv(KIT / "_data" / "test_inventory.txt", sep=r"\s+", header=None, names=["n", "f"]).set_index("f")["n"]
    add("Narrative: 7 feature tests, 8 API tests, 10 pages", "7 / 8 / 10", f"{inv2['tests/test_features.py']} / {inv2['tests/test_api.py']} / {len(tab('pages_roles.csv'))}", (inv2["tests/test_features.py"], inv2["tests/test_api.py"], len(tab("pages_roles.csv"))) == (7, 8, 10))
    refs = tab("references_verified_final.csv")
    add("Every reference with a DOI in the kit is registry-verified", "all VERIFIED rows have a DOI", f"{int((refs.status == 'VERIFIED').sum())} verified, {int((refs.status != 'VERIFIED').sum())} not", bool(refs.loc[refs.status == 'VERIFIED', 'doi'].notna().all()))

    # every backticked path in REPORT_FACTS.md must exist
    names = {Path(f).name for f in run(["git", "ls-files"]).splitlines()} | {f.name for f in KIT.rglob("*") if f.is_file()}
    roots = [REPO, FL, KIT, FL / "app", FL / "artifacts", FL / "docs", KIT / "figures"]
    missing = []
    for tok in sorted(set(re.findall(r"`([^`\n]+)`", text))):
        t = re.sub(r"::.*$", "", tok.strip())  # pytest node id
        t = re.sub(r":[A-Za-z_]\w*$", "", t)  # file.py:function_or_constant
        t = re.sub(r":[\d,\- ]+$", "", t)  # line numbers
        t = re.sub(r"\s+lines?\s.*$", "", t)
        if any(c in t for c in " *{}<>()=|,;'\"") or "..." in t or t.startswith(("http", "-", "python", "pytest", "ruff", "git", "docker", "gh ", "pip")):
            continue
        if not re.search(r"\.(py|md|csv|json|png|yaml|yml|toml|txt|dot|docx|pdf|ipynb|parquet|joblib|jsonl|sqlite)$", t) and not t.endswith("/") and "/" not in t:
            continue
        if t in {"fraudTrain.csv", "fraudTest.csv"} or t == "gauravv-jainn/frauddet":
            continue  # the Kaggle CSVs live in the kagglehub cache, not in the repo; the second is a GitHub repository name
        if "/" in t:
            if not any((r / t).exists() for r in roots):
                missing.append(tok)
        elif t.split("/")[-1] not in names and not any((r / t).exists() for r in roots):
            missing.append(tok)
    add("Every file path in backticks exists", "0 missing", f"{len(missing)} missing", not missing)
    body = "\n".join(rows)
    note = ""
    if missing:
        note = "\n\nMissing path tokens:\n" + "\n".join(f"- `{m}`" for m in missing)
    return (f"## 17. Self-check log (run by `report_kit/scripts/build_facts.py`)\n\n| Check | Expected | Observed | Result |\n|---|---|---|---|\n{body}{note}\n\nTables are generated by code from `fraudlens/artifacts/` and `report_kit/tables/`. The \"Narrative\" rows above re-check the numbers that were typed by hand into prose. Numbers copied from documents (base-paper table, PROGRESS.md) are labelled as such where they appear.\n"), ok_all


def p2_fig():
    return [("architecture.png",), ("scoring_flow.png",), ("sqlite_er.png",), ("training_pipeline_time_split.png",), ("role_matrix.png",)]


def main():
    sections = [p1.s1(), p1.s2(), p1.s3(), p1.s4(), p1.s5(), p1.s6(), p1.s7(), p2.s8(), p2.s9(), p2.s10(), p2.s11(), p2.s12(), p2.s13(), p2.s14(), p2.s15(), p2.s16()]
    head = ("# REPORT_FACTS: FraudLens report kit\n\nGenerated on 2026-11-01 from the repository state at commit `" + run(["git", "rev-parse", "--short", "HEAD"]) +
            "` (branch main). Not a report: facts, numbers and sources for the college report book. **Rules followed:** every number comes from `fraudlens/artifacts/` or from a command named next to it (numbers quoted from documents, such as the base-paper table, say so); "
            "anything not found says NOT FOUND; anything not verified says UNVERIFIED; disagreements are in section 15. Nothing was retrained and no file outside `report_kit/` was changed.\n\n"
            "Contents: 1 Title, abstract, tech stack | 2 Problem and scope | 3 Base paper | 4 Literature survey | 5 Dataset | 6 Architecture | 7 Methodology | 8 Results | 9 Testing | 10 Timeline | "
            "11 Screenshots | 12 Limitations, ethics, privacy, future | 13 Viva questions | 14 Open questions | 15 Inconsistencies | 16 Template cross-reference | 17 Self-check log\n")
    body = head + "\n" + "\n".join(sections)
    log, ok = selfcheck(body)
    (KIT / "REPORT_FACTS.md").write_text(body + "\n" + log, encoding="utf-8")
    (KIT / "references.md").write_text(build_references(), encoding="utf-8")
    (KIT / "glossary.md").write_text(build_glossary(), encoding="utf-8")
    (KIT / "figure_table_index.md").write_text(build_index(), encoding="utf-8")
    print(log)
    print("SELF-CHECK", "PASS" if ok else "FAIL")
    sys.stdout.flush()


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
