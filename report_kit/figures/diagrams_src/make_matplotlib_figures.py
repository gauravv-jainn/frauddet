"""Training pipeline with the time split, and the role/permission matrix.

Numbers come from fraudlens/artifacts/metrics.json and fraudkit.auth.PERMISSIONS (read at run time).
Run from the repo root: fraudlens/.venv/Scripts/python.exe report_kit/figures/diagrams_src/make_matplotlib_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

KIT = Path(__file__).resolve().parents[2]
FL = KIT.parent / "fraudlens"
OUT = KIT / "figures"
sys.path.insert(0, str(FL))
m = json.loads((FL / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
sp = m["split"]

# ---------------------------------------------------------------- training pipeline + time split
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12.5, 7.6), gridspec_kw={"height_ratios": [1.15, 1]})
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Time-based split: the model never sees the future", fontsize=14, weight="bold", loc="left")
n_all = sp["fit"]["rows"] + sp["validation"]["rows"] + sp["test"]["rows"]
parts = [("fit", "#BFDBFE", "Fit (train the model, tune with CV)"), ("validation", "#FDE68A", "Validation (calibration,\nthreshold, rule limits)"), ("test", "#FCA5A5", "Test (used once)")]
x = 2
VAL_W = 17.0  # validation box is drawn wider than its share so its text fits (widths are not to scale)
widths = {"validation": VAL_W}
rest = 96 - VAL_W
widths["fit"] = rest * sp["fit"]["rows"] / (sp["fit"]["rows"] + sp["test"]["rows"])
widths["test"] = rest - widths["fit"]
for key, color, label in parts:
    w = widths[key]
    ax.add_patch(FancyBboxPatch((x, 4.6), w - 0.6, 2.6, boxstyle="round,pad=0.02,rounding_size=0.4", fc=color, ec="#334155"))
    ax.text(x + (w - 0.6) / 2, 6.4, label, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + (w - 0.6) / 2, 5.3, f"{sp[key]['rows']:,} rows | {sp[key]['frauds']:,} frauds\nfraud rate {100 * sp[key]['fraud_rate']:.2f}%", ha="center", va="center", fontsize=9.5)
    ax.text(x, 3.9, sp[key]["start"][:10], fontsize=9, ha="left")
    ax.text(x + w - 0.6, 3.9, sp[key]["end"][:10], fontsize=9, ha="right")
    x += w
ax.annotate("", xy=(98, 8.4), xytext=(2, 8.4), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6))
ax.text(50, 8.9, "time", ha="center", fontsize=10)
ax.text(2, 3.0, "Box widths are not to scale. ", fontsize=9, color="#64748B")
ax.text(2, 2.4, "Per-card features use only EARLIER transactions of the same card; merchant encoding is fitted on the fit split only "
        "(out-of-fold for the fit rows).", fontsize=9.5, color="#334155")
ax.text(2, 1.4, "The fit + validation rows come from the Kaggle file fraudTrain.csv, the test rows from fraudTest.csv (both split by time).", fontsize=9.5, color="#334155")

ax2.set_xlim(0, 100)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("Training pipeline (python -m fraudkit.train), with the measured seconds of the archived 1,236 s run", fontsize=12, weight="bold", loc="left")
steps = [("1 load", "Kaggle CSVs"), ("2 features", "leakage-free"), ("3 split", "fit / val / test"), ("4 baselines", "LogReg, RF,\nLightGBM default"),
         ("5 Optuna", "30 trials, 3-fold CV\nPR-AUC"), ("6 calibration", "isotonic, Platt,\nbeta: log-loss"), ("7 threshold", "min cost on\nvalidation"), ("8 test once", "PR-AUC +\nbootstrap CI"), ("9 save", "atomic,\nversioned")]
run = json.loads(next((FL / "artifacts" / "runs").glob("20261025-233115-*")).joinpath("metrics.json").read_text(encoding="utf-8"))["durations_s"]
secs = [run["load"], run["features"], run["split"], run["baseline"], run["tuning"], run["calibration"], run["threshold"], run["test"], None]
for i, ((t, d), s) in enumerate(zip(steps, secs, strict=True)):
    x0 = 1 + i * 10.9
    ax2.add_patch(FancyBboxPatch((x0, 3.2), 9.6, 4.6, boxstyle="round,pad=0.02,rounding_size=0.4", fc="#F1F5F9", ec="#475569"))
    ax2.text(x0 + 4.8, 7.0, t, ha="center", va="center", fontsize=9.5, weight="bold")
    ax2.text(x0 + 4.8, 5.4, d, ha="center", va="center", fontsize=8.4)
    ax2.text(x0 + 4.8, 3.8, "" if s is None else f"{s:,.1f} s", ha="center", va="center", fontsize=9, color="#1D4ED8")
    if i < len(steps) - 1:
        ax2.annotate("", xy=(x0 + 10.7, 5.5), xytext=(x0 + 9.7, 5.5), arrowprops=dict(arrowstyle="->", color="#475569"))
ax2.text(1, 1.6, "Durations: fraudlens/artifacts/runs/20261025-233115-3abe0ebd/metrics.json -> durations_s (the final run's wall-clock "
         "68,989 s includes the computer sleeping).", fontsize=8.8, color="#334155")
fig.tight_layout()
fig.savefig(OUT / "training_pipeline_time_split.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- role / permission matrix
from fraudkit.auth import PERMISSIONS, ROLES  # noqa: E402

perms = list(PERMISSIONS)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.set_xlim(0, len(ROLES) + 3.2)
ax.set_ylim(0, len(perms) + 1)
ax.axis("off")
ax.set_title("Role / permission matrix (fraudkit/auth.py PERMISSIONS)", fontsize=13, weight="bold", loc="left")
for j, r in enumerate(ROLES):
    ax.text(3.6 + j, len(perms) + 0.3, r, ha="center", va="center", weight="bold", fontsize=11)
for i, p in enumerate(perms):
    y = len(perms) - 1 - i + 0.3
    ax.text(0.1, y, p.replace("_", " "), va="center", fontsize=10.5)
    for j, r in enumerate(ROLES):
        ok = r in PERMISSIONS[p]
        ax.add_patch(FancyBboxPatch((3.15 + j, y - 0.3), 0.9, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#BBF7D0" if ok else "#F1F5F9", ec="#94A3B8"))
        ax.text(3.6 + j, y, "yes" if ok else "-", ha="center", va="center", fontsize=10, color="#166534" if ok else "#94A3B8")
fig.tight_layout()
fig.savefig(OUT / "role_matrix.png", dpi=170)
print("ok")
