"""Print the README results section from artifacts/metrics.json (so no number is typed by hand)."""
import json
from pathlib import Path

m = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
t, a, c, tn = m["test"], m["test"]["at_threshold"], m["calibration"], m["tuning"]

print("**Validation leaderboard** (last 15% of the training period, "
      f"{m['split']['validation']['rows']:,} transactions, {m['split']['validation']['frauds']:,} frauds)\n")
print("| Model | PR-AUC | ROC-AUC | Brier | Fit time (s) |\n|---|---|---|---|---|")
for r in m["leaderboard"]:
    name = f"**{r['model']}** (selected)" if r["selected"] else r["model"]
    print(f"| {name} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['brier']:.5f} | {r['fit_seconds']:.1f} |")
print(f"\nOptuna: {tn['trials']} trials, {tn['cv_folds']}-fold stratified CV, best CV PR-AUC {tn['best_cv_pr_auc']:.4f}.\n")
print(f"**Test set** ({a['transactions']:,} transactions from {m['split']['test']['start'][:10]} to "
      f"{m['split']['test']['end'][:10]}, {m['split']['test']['frauds']:,} frauds), evaluated once:\n")
print("| Metric | Value |\n|---|---|")
rows = [
    ("PR-AUC (95% bootstrap CI)", f"{t['pr_auc']:.4f} ({t['pr_auc_ci95'][0]:.4f} - {t['pr_auc_ci95'][1]:.4f}), "
                                  f"{t['bootstrap_resamples']} resamples"),
    ("ROC-AUC", f"{t['roc_auc']:.4f}"),
    ("Brier score, uncalibrated -> calibrated", f"{t['brier_before_calibration']:.5f} -> {t['brier_after_calibration']:.5f} "
                                                f"({c['method']})"),
    ("Decision threshold (chosen on validation)", f"{a['threshold']:.4f} (review cost {a['review_cost']:.0f} per alert)"),
    ("Recall at threshold", f"{a['recall']:.1%}"),
    ("Precision at threshold", f"{a['precision']:.1%}"),
    ("False alarms per 10,000 genuine", f"{a['false_alarms_per_10k_genuine']:.1f}"),
    ("Transactions flagged", f"{a['flagged']:,}"),
    ("Fraud amount in test period", f"{a['fraud_amount']:,.0f}"),
    ("Cost with model (missed fraud + reviews)", f"{a['missed_amount']:,.0f} + {a['review_cost_total']:,.0f} = "
                                                 f"{a['total_cost']:,.0f}"),
    ("**Saved vs no model**", f"**{a['saved_vs_no_model']:,.0f}**"),
]
for k, v in rows:
    print(f"| {k} | {v} |")
print(f"\nFull training run: {m['run']['total_seconds']:.0f} s on {m['machine']['cpu_count']} CPU threads "
      f"({m['machine']['platform']}).")
