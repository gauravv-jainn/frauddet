"""Write report_kit/code_snippets.md: verbatim excerpts (<= 25 lines) cut from the source files by line range."""

from __future__ import annotations

import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
REPO = KIT.parent

# (repo-relative path, first line, last line, title, what it shows / why it matters for the report)
SNIPPETS = [
    ("fraudlens/fraudkit/features.py", 147, 165, "Rows are ordered so a row never sees a later row",
     "Leakage prevention: rows are sorted by (card, time, original position); every card feature below is computed on earlier rows only."),
    ("fraudlens/fraudkit/features.py", 92, 99, "Rolling window count and sum over EARLIER rows only",
     "`count = idx - lo` excludes the current row (`lo..i-1`); the same helper builds the 1 h / 24 h / 7 d / 30 d velocity features."),
    ("fraudlens/fraudkit/features.py", 128, 135, "Merchant / category risk with a 7-day label delay",
     "Fraud labels younger than 7 days are ignored (chargebacks arrive late in production); the rate is smoothed towards a 0.5% prior."),
    ("fraudlens/fraudkit/features.py", 311, 320, "Out-of-fold merchant target encoding",
     "A fit row never sees its own label through the merchant feature: each fold is encoded with a mapping fitted on the other folds."),
    ("fraudlens/fraudkit/train.py", 349, 358, "Time-based fit / validation split and encoder fitting",
     "The validation period is the LAST 15% of the training file by time; the encoder is fitted on the fit rows only."),
    ("fraudlens/fraudkit/train.py", 460, 477, "Optuna objective (3-fold CV PR-AUC)",
     "The 10 tuned LightGBM hyper-parameters and their search ranges; the objective is mean PR-AUC over the folds. Folds are stratified and shuffled (not time-ordered)."),
    ("fraudlens/fraudkit/score.py", 61, 85, "Calibrator: Platt / beta / isotonic",
     "Three calibration methods behind one class; outputs are clipped to [1e-12, 1 - 1e-12] so a probability is never exactly 0 or 1."),
    ("fraudlens/fraudkit/score.py", 29, 39, "0-999 risk score on a log-odds scale",
     "p = 1e-6 maps to 0 and p = 0.999 to 999; `score_points_to_double_odds` gives the points that double the odds."),
    ("fraudlens/fraudkit/decision.py", 33, 52, "Decision policy: flags and the four actions",
     "Threshold mode vs expected-loss mode (p x amount >= k x review cost) and the approve / step_up / review / block logic."),
    ("fraudlens/fraudkit/rules.py", 87, 111, "Policy rules (guardrail on top of the model)",
     "Vectorised rules (large amount, spend spike, far from home, impossible travel, burst, out-of-range, watchlists); they can only raise a band."),
    ("fraudlens/fraudkit/explain.py", 347, 364, "From SHAP values to a reason-coded explanation",
     "One call: features -> raw score -> calibrated p -> band -> risk score -> SHAP contributions -> reasons -> policy rule hits."),
    ("fraudlens/fraudkit/monitoring.py", 39, 50, "Population Stability Index and its status bands",
     "PSI = sum((actual - expected) * ln(actual / expected)); < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant."),
    ("fraudlens/fraudkit/auth.py", 60, 72, "Password hashing (PBKDF2-HMAC-SHA256, 600,000 iterations)",
     "Salted PBKDF2 with a constant-time comparison; the iteration count constant is `PBKDF2_ITERATIONS` in the same file."),
    ("fraudlens/api/main.py", 270, 290, "POST /v1/score: validate, then featurize with the card's history",
     "Batch-size limit (413), schema validation (422), then features are built from the request rows plus earlier rows of the same cards."),
    ("fraudlens/fraudkit/db.py", 123, 135, "Versioned SQLite migrations",
     "`PRAGMA user_version` tracks the schema; an old database is upgraded in place, each migration inside one transaction."),
]

out = [
    "# Code snippets (verbatim)\n",
    "Every excerpt below is cut from the file by line range by `report_kit/scripts/make_snippets.py` (no retyping). "
    f"Line numbers refer to the repository state at commit `{__import__('subprocess').run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()}` "  # noqa: S603, S607
    "plus the uncommitted report_kit/ files. Each excerpt is at most 25 lines.\n",
]
for i, (rel, a, b, title, why) in enumerate(SNIPPETS, 1):
    lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
    assert 1 <= a <= b <= len(lines) and b - a + 1 <= 25, (rel, a, b)
    body = "\n".join(lines[a - 1 : b])
    out.append(f"## S{i}. {title}\n\n`{rel}` lines {a}-{b}. {why}\n\n```python\n{body}\n```\n")
(KIT / "code_snippets.md").write_text("\n".join(out), encoding="utf-8")
print(f"wrote {len(SNIPPETS)} snippets")
sys.stdout.flush()
