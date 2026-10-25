"""Train FraudLens end to end with a live terminal view.

    python -m fraudkit.train --config configs/default.yaml           # full run
    python -m fraudkit.train --config configs/default.yaml --fast    # ~2 minute smoke run

Stages: load, features, split, baseline, tuning, calibration, threshold, test, save.
Every step is appended to <out>/training_log.jsonl; Optuna history goes to <out>/trials.csv.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from .security import atomic_path, atomic_write_bytes, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts"
STAGES = ["load", "features", "split", "baseline", "tuning", "calibration", "threshold", "test", "save"]
CPU = os.cpu_count() or 1


# --------------------------------------------------------------------------- logging + live view
class RunLog:
    def __init__(self, out: Path):
        self.path = out / "training_log.jsonl"
        self.path.write_text("", encoding="utf-8")  # one log per run (latest run)
        self.t0 = time.time()

    def log(self, stage: str, metric: str, value=None, **extra):
        rec = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "elapsed_s": round(time.time() - self.t0, 3),
            "stage": stage,
            "metric": metric,
            "value": value,
        }
        rec.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class LiveView:
    """Rich live display: stage list with progress bars + live Optuna table."""

    def __init__(self, enabled: bool):
        from rich.console import Console
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        self.console = Console()
        self.enabled = enabled and self.console.is_terminal
        self.narrow = self.console.width < 100
        cols = [
            SpinnerColumn(),
            TextColumn("[bold]{task.description:<11}"),
            BarColumn(bar_width=12 if self.narrow else 30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]
        if not self.narrow:
            cols.insert(4, TextColumn("{task.fields[note]}"))
        self.progress = Progress(*cols, console=self.console)
        self.tasks = {}
        self.trials: list[dict] = []
        self.best = None
        self.live = None

    def __enter__(self):
        if self.enabled:
            from rich.live import Live

            self.live = Live(get_renderable=self.render, console=self.console, refresh_per_second=4)
            self.live.__enter__()
        return self

    def __exit__(self, *exc):
        if self.live:
            self.live.__exit__(*exc)

    def render(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        parts = [Panel(self.progress, title="FraudLens training", border_style="cyan")]
        if self.trials:
            from rich import box

            t = Table(title="Optuna: PR-AUC per trial (CV)", box=box.SIMPLE_HEAD, padding=(0, 1), expand=False)
            cols = ["#", "PR-AUC", "best", "lr", "leaves", "trees"] + (
                [] if self.narrow else ["min_child", "spw", "sec"]
            )
            for c in cols:
                t.add_column(c, justify="right", no_wrap=True)
            for r in self.trials[-12:]:
                style = "bold green" if r["value"] == r["best"] else None
                p = r["params"]
                cells = [
                    str(r["number"]),
                    f"{r['value']:.4f}",
                    f"{r['best']:.4f}",
                    f"{p['learning_rate']:.3f}",
                    str(p["num_leaves"]),
                    str(p["n_estimators"]),
                ]
                if not self.narrow:
                    cells += [str(p["min_child_samples"]), f"{p['scale_pos_weight']:.1f}", f"{r['seconds']:.1f}"]
                t.add_row(*cells, style=style)
            parts.append(t)
        return Group(*parts)

    def start_stage(self, name: str, total: int):
        self.tasks[name] = self.progress.add_task(name, total=max(total, 1), note="")
        if not self.enabled:
            self.console.print(f"[stage] {name} ...")

    def advance(self, name: str, note: str = ""):
        self.progress.update(self.tasks[name], advance=1, note=note)
        if not self.enabled and note:
            self.console.print(f"  {name}: {note}")

    def finish_stage(self, name: str, note: str = ""):
        if note and self.enabled:
            self.console.print(f"[green]done[/] [bold]{name}[/]: {note}")
        t = self.tasks[name]
        total = next(x.total for x in self.progress.tasks if x.id == t)
        self.progress.update(t, completed=total, note=note)
        if not self.enabled:
            self.console.print(f"[done] {name} {note}")

    def print(self, *a, **k):
        self.console.print(*a, **k)


class Runner:
    def __init__(self, out: Path, live: LiveView):
        self.out = out
        self.log = RunLog(out)
        self.view = live
        self.durations: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str, total: int = 1):
        self.view.start_stage(name, total)
        self.log.log(name, "stage_start")
        t0 = time.time()
        box = {"note": ""}
        yield box
        dt = time.time() - t0
        self.durations[name] = round(dt, 2)
        self.log.log(name, "stage_end", round(dt, 3))
        self.view.finish_stage(name, box["note"])


# --------------------------------------------------------------------------- helpers
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def lib_versions() -> dict:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for p in ["pandas", "numpy", "scikit-learn", "lightgbm", "shap", "optuna", "streamlit", "plotly"]:
        with contextlib.suppress(md.PackageNotFoundError):
            out[p] = md.version(p)
    return out


def machine_info() -> dict:
    return {
        "cpu_count": CPU,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def average_precision_weighted(y_sorted: np.ndarray, ends: np.ndarray, w: np.ndarray) -> float:
    """sklearn-equivalent average precision on pre-sorted scores with sample weights."""
    tps = np.cumsum(w * y_sorted)[ends]
    fps = np.cumsum(w * (1 - y_sorted))[ends]
    if tps[-1] == 0:
        return float("nan")
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def bootstrap_pr_auc(y, p, n_boot: int, seed: int) -> tuple[float, float, float]:
    y = np.asarray(y, dtype="float64")
    order = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[order]
    ps = np.asarray(p)[order]
    ends = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
    point = average_precision_weighted(ys, ends, np.ones_like(ys))
    rng = np.random.default_rng(seed)
    n = len(ys)
    vals = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype("float64")
        vals.append(average_precision_weighted(ys, ends, w))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


RELIABILITY_EDGES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def reliability_table(y, p) -> list[dict]:
    """Reliability table on fixed probability bins (empty bins are skipped).

    Fixed bins rather than equal-count bins: with 0.4% fraud, equal-count bins all sit near zero.
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype="float64")})
    df["bin"] = pd.cut(df["p"], bins=RELIABILITY_EDGES, include_lowest=True, right=False)
    df.loc[df["p"] >= 1.0, "bin"] = df["bin"].cat.categories[-1]
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), frauds=("y", "sum"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    )
    g["bin"] = [f"{iv.left:.2f}-{iv.right:.2f}" for iv in g.index]
    return g.reset_index(drop=True).to_dict(orient="records")


def subsample_negatives(idx: np.ndarray, y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    keep = rng.choice(neg, size=int(len(neg) * frac), replace=False)
    return np.sort(np.r_[pos, keep])


def lgbm_base(seed: int) -> dict:
    return {"random_state": seed, "n_jobs": CPU, "deterministic": True, "force_col_wise": True, "verbose": -1}


# --------------------------------------------------------------------------- main pipeline
def run(cfg: dict, out: Path, fast: bool, trials: int | None, sample_cards: int | None, live_enabled: bool) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    from . import data as D
    from .costs import best_threshold, confusion_at, cost_curve, downsample_curve
    from .features import CATEGORICAL, CONTEXT, FEATURES, NUMERIC, MerchantEncoder, build_features, model_frame
    from .decision import policy_flags, tune_expected_loss
    from .monitoring import reference_profile
    from .score import Calibrator, FraudModel

    seed = int(cfg["seed"])
    seed_everything(seed)
    out.mkdir(parents=True, exist_ok=True)
    n_trials = trials if trials is not None else (3 if fast else int(cfg["model"]["optuna_trials"]))
    if fast and sample_cards is None:
        sample_cards = 100
    rf_trees = 40 if fast else 200
    n_boot = 200 if fast else 1000
    neg_frac = float(cfg["model"]["tuning_sample_negatives"])
    review_cost = float(cfg["costs"]["review_cost"])
    t_start = time.time()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    view = LiveView(live_enabled)
    R = Runner(out, view)
    status_path = out / "run_status.json"
    atomic_write_text(
        status_path,
        json.dumps({"state": "running", "pid": os.getpid(), "started": started_at, "fast": fast, "trials": n_trials}),
    )
    R.log.log(
        "run",
        "config",
        {
            "fast": fast,
            "trials": n_trials,
            "sample_cards": sample_cards,
            "seed": seed,
            "review_cost": review_cost,
            "cpu_count": CPU,
        },
    )
    M: dict = {
        "run": {
            "started_at": started_at,
            "fast": fast,
            "trials": n_trials,
            "sample_cards": sample_cards,
            "seed": seed,
            "config": cfg,
        }
    }

    with view:
        # ---------------------------------------------------------------- load
        with R.stage("load", 2) as st:
            raw = D.load_raw(cfg["data"].get("cache_dir"), sample_cards=sample_cards, seed=seed)
            view.advance("load", "read CSVs")
            dtr, dte = D.describe(raw.train), D.describe(raw.test)
            for k, v in dtr.items():
                R.log.log("load", f"train_{k}", v)
            for k, v in dte.items():
                R.log.log("load", f"test_{k}", v)
            view.advance("load")
            st["note"] = f"train {dtr['rows']:,} rows, test {dte['rows']:,} rows"
            M["data"] = {"source": raw.source, "train": dtr, "test": dte}

        # ---------------------------------------------------------------- features
        with R.stage("features", 1) as st:
            combo = pd.concat([raw.train.assign(_split="train"), raw.test.assign(_split="test")], ignore_index=True)
            feat = build_features(combo)
            view.advance("features")
            st["note"] = f"{len(FEATURES)} features on {len(feat):,} rows"
            R.log.log("features", "n_features", len(FEATURES))

        # ---------------------------------------------------------------- split
        with R.stage("split", 1) as st:
            tr_all = feat[feat["_split"] == "train"].sort_values("trans_date_trans_time", kind="mergesort")
            test = feat[feat["_split"] == "test"]
            cut_i = int(len(tr_all) * (1 - float(cfg["data"]["validation_fraction"])))
            cutoff = tr_all["trans_date_trans_time"].iloc[cut_i]
            fit = tr_all[tr_all["trans_date_trans_time"] < cutoff].copy()
            val = tr_all[tr_all["trans_date_trans_time"] >= cutoff].copy()
            test = test.copy()
            enc = MerchantEncoder(smoothing=float(cfg["features"]["te_smoothing"]), seed=seed)
            fit["merchant_te"] = enc.fit_transform_oof(fit["merchant"], fit["is_fraud"])
            val["merchant_te"] = enc.transform(val["merchant"])
            test["merchant_te"] = enc.transform(test["merchant"])
            cats = {c: sorted(fit[c].dropna().unique().tolist()) for c in CATEGORICAL}
            X_fit, X_val, X_test = (model_frame(d, cats) for d in (fit, val, test))
            y_fit, y_val, y_test = (d["is_fraud"].to_numpy().astype(int) for d in (fit, val, test))
            split_info = {}
            for name, d in [("fit", fit), ("validation", val), ("test", test)]:
                split_info[name] = {
                    "rows": int(len(d)),
                    "frauds": int(d["is_fraud"].sum()),
                    "fraud_rate": float(d["is_fraud"].mean()),
                    "start": str(d["trans_date_trans_time"].min()),
                    "end": str(d["trans_date_trans_time"].max()),
                }
                R.log.log("split", f"{name}_rows", int(len(d)))
                R.log.log("split", f"{name}_frauds", int(d["is_fraud"].sum()))
            M["split"] = split_info
            view.advance("split")
            st["note"] = f"fit {len(fit):,} / val {len(val):,} / test {len(test):,}"

        leaderboard = []

        def evaluate(name, p_val, fit_s, extra=None):
            row = {
                "model": name,
                "pr_auc": float(average_precision_score(y_val, p_val)),
                "roc_auc": float(roc_auc_score(y_val, p_val)),
                "brier": float(brier_score_loss(y_val, p_val)),
                "fit_seconds": round(fit_s, 2),
            }
            row.update(extra or {})
            leaderboard.append(row)
            for k in ["pr_auc", "roc_auc", "brier"]:
                R.log.log("baseline" if "tuned" not in name else "tuning", f"{name}:{k}", row[k])
            return row

        # ---------------------------------------------------------------- baselines
        models = {}
        with R.stage("baseline", 3) as st:
            # Logistic regression: log1p on non-negative numerics, impute, scale, one-hot categories.
            lr = make_pipeline(
                ColumnTransformer(
                    [
                        (
                            "num",
                            make_pipeline(
                                FunctionTransformer(
                                    lambda a: np.log1p(np.clip(a, 0, None)), feature_names_out="one-to-one"
                                ),
                                SimpleImputer(strategy="median", add_indicator=True),
                                StandardScaler(),
                            ),
                            NUMERIC,
                        ),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                    ]
                ),
                LogisticRegression(max_iter=2000, C=1.0),
            )
            t0 = time.time()
            lr.fit(X_fit, y_fit)
            r = evaluate("Logistic regression", lr.predict_proba(X_val)[:, 1], time.time() - t0)
            view.advance("baseline", f"LogReg PR-AUC {r['pr_auc']:.4f}")

            def to_numeric(X):
                Z = X.copy()
                for c in CATEGORICAL:
                    Z[c] = Z[c].cat.codes
                return Z.fillna(-1).to_numpy(dtype="float32")

            rf = RandomForestClassifier(
                n_estimators=rf_trees,
                min_samples_leaf=5,
                max_features="sqrt",
                max_samples=0.3,
                n_jobs=-1,
                random_state=seed,
            )
            t0 = time.time()
            rf.fit(to_numeric(X_fit), y_fit)
            r = evaluate("Random forest", rf.predict_proba(to_numeric(X_val))[:, 1], time.time() - t0)
            view.advance("baseline", f"RF PR-AUC {r['pr_auc']:.4f}")

            t0 = time.time()
            lgb_default = LGBMClassifier(**lgbm_base(seed)).fit(X_fit, y_fit)
            r = evaluate("LightGBM default", lgb_default.predict_proba(X_val)[:, 1], time.time() - t0)
            models["LightGBM default"] = lgb_default
            view.advance("baseline", f"LGBM PR-AUC {r['pr_auc']:.4f}")
            st["note"] = "3 models scored on validation"

        # ---------------------------------------------------------------- tuning
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        skf = StratifiedKFold(n_splits=int(cfg["model"]["cv_folds"]), shuffle=True, random_state=seed)
        folds = [
            (subsample_negatives(tr, y_fit, neg_frac, seed + k), te)
            for k, (tr, te) in enumerate(skf.split(np.zeros(len(y_fit)), y_fit))
        ]
        trial_rows = []

        def objective(trial: optuna.Trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 1200, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
            }
            scores = []
            for tr, te in folds:
                m = LGBMClassifier(**params, **lgbm_base(seed)).fit(X_fit.iloc[tr], y_fit[tr])
                scores.append(average_precision_score(y_fit[te], m.predict_proba(X_fit.iloc[te])[:, 1]))
            return float(np.mean(scores))

        with R.stage("tuning", n_trials + 1) as st:

            def on_trial(study, trial):
                best = study.best_value
                row = {
                    "number": trial.number,
                    "value": trial.value,
                    "best": best,
                    "seconds": (trial.datetime_complete - trial.datetime_start).total_seconds(),
                    "params": trial.params,
                }
                view.trials.append(row)
                trial_rows.append(
                    {
                        "trial": trial.number,
                        "pr_auc_cv": trial.value,
                        "best_so_far": best,
                        "seconds": row["seconds"],
                        **trial.params,
                    }
                )
                R.log.log(
                    "tuning",
                    "trial_pr_auc",
                    trial.value,
                    trial=trial.number,
                    best_so_far=best,
                    params=trial.params,
                    seconds=row["seconds"],
                )
                view.advance("tuning", f"trial {trial.number}: {trial.value:.4f} (best {best:.4f})")

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=n_trials, callbacks=[on_trial])
            atomic_write_text(out / "trials.csv", pd.DataFrame(trial_rows).to_csv(index=False))
            best_params = dict(study.best_params)
            R.log.log("tuning", "best_cv_pr_auc", study.best_value, params=best_params)
            fit_idx = subsample_negatives(np.arange(len(y_fit)), y_fit, neg_frac, seed)
            t0 = time.time()
            lgb_tuned = LGBMClassifier(**best_params, subsample_freq=1, **lgbm_base(seed)).fit(
                X_fit.iloc[fit_idx], y_fit[fit_idx]
            )
            r = evaluate(
                "LightGBM tuned (Optuna)",
                lgb_tuned.predict_proba(X_val)[:, 1],
                time.time() - t0,
                {"cv_pr_auc": float(study.best_value)},
            )
            models["LightGBM tuned (Optuna)"] = lgb_tuned
            view.advance("tuning", f"tuned model val PR-AUC {r['pr_auc']:.4f}")
            st["note"] = f"best CV PR-AUC {study.best_value:.4f}"
            M["tuning"] = {
                "trials": n_trials,
                "best_cv_pr_auc": float(study.best_value),
                "best_params": best_params,
                "cv_folds": int(cfg["model"]["cv_folds"]),
                "fit_negative_fraction": neg_frac,
            }

        leaderboard.sort(key=lambda r: -r["pr_auc"])
        lgb_rows = [r for r in leaderboard if r["model"] in models]
        chosen = lgb_rows[0]["model"]
        model = models[chosen]
        for r in leaderboard:
            r["selected"] = r["model"] == chosen
        M["leaderboard"] = leaderboard
        M["selected_model"] = chosen
        R.log.log("tuning", "selected_model", chosen)

        # ---------------------------------------------------------------- calibration
        with R.stage("calibration", 1) as st:
            p_val_raw = model.predict_proba(X_val)[:, 1]
            n_pos_val = int(y_val.sum())
            method = "isotonic" if n_pos_val >= 300 else "platt"
            cal = Calibrator(method).fit(p_val_raw, y_val)
            p_val = cal.transform(p_val_raw)
            M["calibration"] = {
                "method": method,
                "validation_positives": n_pos_val,
                "validation_brier_before": float(brier_score_loss(y_val, p_val_raw)),
                "validation_brier_after": float(brier_score_loss(y_val, p_val)),
                "validation_reliability_before": reliability_table(y_val, p_val_raw),
                "validation_reliability_after": reliability_table(y_val, p_val),
            }
            R.log.log("calibration", "method", method)
            R.log.log("calibration", "val_brier_before", M["calibration"]["validation_brier_before"])
            R.log.log("calibration", "val_brier_after", M["calibration"]["validation_brier_after"])
            view.advance("calibration")
            st["note"] = (
                f"{method}: Brier {M['calibration']['validation_brier_before']:.5f} -> "
                f"{M['calibration']['validation_brier_after']:.5f}"
            )

        # ---------------------------------------------------------------- threshold
        with R.stage("threshold", 1) as st:
            thr, curve_val = best_threshold(y_val, p_val, val["amt"].to_numpy(), review_cost)
            at_val = confusion_at(y_val, p_val, val["amt"].to_numpy(), thr, review_cost)
            M["threshold"] = {
                "value": thr,
                "review_cost": review_cost,
                "chosen_on": "validation",
                "validation_at_threshold": at_val,
                "validation_curve": downsample_curve(curve_val).to_dict(orient="list"),
            }
            R.log.log("threshold", "threshold", thr)
            R.log.log("threshold", "val_total_cost", at_val["total_cost"])
            R.log.log("threshold", "val_recall", at_val["recall"])
            # expected-loss policy: review when p x amount >= k x review cost; k tuned on validation
            critical = float(cfg["bands"]["critical"])
            amt_val = val["amt"].to_numpy()
            best_k, k_table = tune_expected_loss(y_val, p_val, amt_val, review_cost, critical)
            el_pol = {"mode": "expected_loss", "k": best_k, "review_cost": review_cost, "critical": critical}
            el_val = confusion_at(y_val, policy_flags(p_val, amt_val, el_pol).astype(float), amt_val, 0.5, review_cost)
            thr_pol = {"mode": "threshold", "threshold": thr, "review_cost": review_cost, "critical": critical}
            thr_val = confusion_at(
                y_val, policy_flags(p_val, amt_val, thr_pol).astype(float), amt_val, 0.5, review_cost
            )
            served = "expected_loss" if el_val["total_cost"] < thr_val["total_cost"] else "threshold"
            M["decision"] = {
                "served": served,
                "chosen_on": "validation (lower total cost)",
                "threshold": {"value": thr, "validation": thr_val},
                "expected_loss": {"k": best_k, "validation": el_val, "k_grid": k_table},
            }
            R.log.log("threshold", "expected_loss_k", best_k)
            R.log.log("threshold", "val_cost_expected_loss", el_val["total_cost"])
            R.log.log("threshold", "val_cost_threshold_policy", thr_val["total_cost"])
            R.log.log("threshold", "served_policy", served)

            # policy-rule limits from the data: amount and distance from the training data, spike rule on validation
            fit_fraud_max = float(fit.loc[fit["is_fraud"] == 1, "amt"].max())
            review_amount = float(np.ceil(1.25 * fit_fraud_max / 500.0) * 500.0)
            far_km = float(np.ceil(2.0 * float(fit["dist_home_km"].max()) / 100.0) * 100.0)
            served_pol = el_pol if served == "expected_loss" else thr_pol
            base_flags = policy_flags(p_val, amt_val, served_pol)
            spike_rows = []
            ratio_val = val["amt_to_card_mean"].to_numpy()
            for ratio in (10.0, 15.0, 20.0, 30.0, 50.0):
                for min_amt in (500.0, 1000.0, 1500.0, 2000.0):
                    hit = (ratio_val >= ratio) & (amt_val >= min_amt)
                    mm = confusion_at(y_val, (base_flags | hit).astype(float), amt_val, 0.5, review_cost)
                    spike_rows.append(
                        {
                            "ratio": ratio,
                            "min_amount": min_amt,
                            "hits": int(hit.sum()),
                            "extra_flags": int((hit & ~base_flags).sum()),
                            "extra_frauds": int((hit & ~base_flags & (y_val == 1)).sum()),
                            "total_cost": mm["total_cost"],
                        }
                    )
            best_spike = min(spike_rows, key=lambda r: (r["total_cost"], -r["ratio"], -r["min_amount"]))
            M["rules_auto"] = {
                "review_amount": review_amount,
                "far_from_home_km": far_km,
                "spend_spike_ratio": best_spike["ratio"],
                "spend_spike_min_amount": best_spike["min_amount"],
                "derivation": {
                    "review_amount": (
                        f"1.25 x largest fraud amount in the fit period ({fit_fraud_max:,.2f}), rounded up to 500"
                    ),
                    "far_from_home_km": "2 x largest home-merchant distance in the fit period, rounded up to 100 km",
                    "spend_spike": "grid search on validation, lowest total cost of policy + rule (ties -> stricter)",
                },
                "spend_spike_grid": spike_rows,
            }
            R.log.log(
                "threshold",
                "rules_auto",
                {
                    k: M["rules_auto"][k]
                    for k in ("review_amount", "far_from_home_km", "spend_spike_ratio", "spend_spike_min_amount")
                },
            )
            view.advance("threshold")
            st["note"] = (
                f"threshold {thr:.4f}, EL k={best_k:g}; served {served} "
                f"(val cost {min(el_val['total_cost'], thr_val['total_cost']):,.0f} vs "
                f"{at_val['cost_no_model']:,.0f} without a model)"
            )

        # ---------------------------------------------------------------- test (once)
        with R.stage("test", 3) as st:
            p_test_raw = model.predict_proba(X_test)[:, 1]
            p_test = cal.transform(p_test_raw)
            pr, lo, hi = bootstrap_pr_auc(y_test, p_test, n_boot, seed)
            view.advance("test", f"PR-AUC {pr:.4f} [{lo:.4f}, {hi:.4f}]")
            pr_raw = float(average_precision_score(y_test, p_test_raw))
            at_test = confusion_at(y_test, p_test, test["amt"].to_numpy(), thr, review_cost)
            amt_test = test["amt"].to_numpy()
            for name, pol in (("threshold", thr_pol), ("expected_loss", el_pol)):
                M["decision"][name]["test"] = confusion_at(
                    y_test, policy_flags(p_test, amt_test, pol).astype(float), amt_test, 0.5, review_cost
                )
            at_policy = M["decision"][served]["test"]
            curve_test = cost_curve(y_test, p_test, test["amt"].to_numpy(), review_cost)
            view.advance("test")
            M["test"] = {
                "pr_auc": pr,
                "pr_auc_ci95": [lo, hi],
                "bootstrap_resamples": n_boot,
                "pr_auc_uncalibrated": pr_raw,
                "roc_auc": float(roc_auc_score(y_test, p_test)),
                "brier_before_calibration": float(brier_score_loss(y_test, p_test_raw)),
                "brier_after_calibration": float(brier_score_loss(y_test, p_test)),
                "reliability_before": reliability_table(y_test, p_test_raw),
                "reliability_after": reliability_table(y_test, p_test),
                "at_threshold": at_test,
                "at_policy": at_policy,
                "curve": downsample_curve(curve_test).to_dict(orient="list"),
            }
            for k in ["pr_auc", "roc_auc", "brier_after_calibration"]:
                R.log.log("test", k, M["test"][k])
            R.log.log("test", "pr_auc_ci_low", lo)
            R.log.log("test", "pr_auc_ci_high", hi)
            for k in ["recall", "precision", "false_alarms_per_10k_genuine", "saved_vs_no_model", "flagged"]:
                R.log.log("test", k, at_test[k])
            view.advance("test")
            st["note"] = (
                f"recall {at_test['recall']:.3f}, saved {at_test['saved_vs_no_model']:,.0f} "
                f"of {at_test['fraud_amount']:,.0f}"
            )

        # ---------------------------------------------------------------- save
        with R.stage("save", 3) as st:
            import warnings

            import shap

            warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")

            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_val), size=min(5000, len(X_val)), replace=False)
            sv = shap.TreeExplainer(model).shap_values(X_val.iloc[np.sort(idx)])
            sv = sv[1] if isinstance(sv, list) else sv
            imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
            M["global_importance"] = {"sample": "validation (5,000 random rows)", "mean_abs_shap": imp.to_dict()}
            view.advance("save", "global SHAP")

            finished = datetime.now(UTC)
            fm = FraudModel(
                model=model,
                calibrator=cal,
                encoder=enc,
                categories=cats,
                threshold=thr,
                bands={k: cfg["bands"].get(k) for k in ["medium", "high", "critical"]},
                review_cost=review_cost,
                model_name=chosen,
                data_source=raw.source,
                versions=lib_versions(),
            )
            # Stage every artifact in a temp file first, then swap them in with os.replace, so the app or the
            # API never reads a half-written model while a retrain is finishing.
            staged: list[tuple[Path, Path]] = []

            def stage(name: str, writer):
                tmp = atomic_path(out / name)
                writer(tmp)
                staged.append((tmp, out / name))

            model_tmp = atomic_path(out / "model.joblib")
            fm.save(model_tmp)
            digest = hashlib.sha256(model_tmp.read_bytes()).hexdigest()[:8]
            version = f"{finished:%Y%m%d-%H%M%S}-{digest}"
            fm.version = version
            fm.save(model_tmp)  # re-save with the version embedded
            staged.append((model_tmp, out / "model.joblib"))

            M["run"]["model_version"] = version
            M["versions"] = lib_versions()
            M["machine"] = machine_info()
            M["features"] = FEATURES
            M["durations_s"] = dict(R.durations)
            M["run"]["finished_at"] = finished.isoformat(timespec="seconds")
            M["run"]["total_seconds"] = round(time.time() - t_start, 1)

            profile = reference_profile(X_fit, NUMERIC, CATEGORICAL, scores=p_val)
            profile["model_version"] = version
            profile["reference"] = "fit period (features), validation period (score)"
            stage(
                "reference_profile.json",
                lambda t: t.write_text(json.dumps(profile, default=_json_default), encoding="utf-8"),
            )

            # scored test set for the app (gitignored), exactly the numbers in metrics.json
            keep = [
                "trans_num",
                "trans_date_trans_time",
                "cc_num",
                "merchant",
                "category",
                "amt",
                "gender",
                "city",
                "state",
                "lat",
                "long",
                "city_pop",
                "job",
                "dob",
                "merch_lat",
                "merch_long",
                "is_fraud",
            ]
            ctx = list(CONTEXT)
            scored = test[keep + [c for c in FEATURES if c not in keep] + ctx].copy()
            scored["p_raw"] = p_test_raw
            scored["p"] = p_test
            stage("test_scored.parquet", lambda t: scored.reset_index(drop=True).to_parquet(t, index=False))
            hist = combo.drop(columns=["_split"])
            stage("history.parquet", lambda t: hist.to_parquet(t, index=False))
            view.advance("save", "model + scored test set")
            metrics_text = json.dumps(M, indent=2, default=_json_default)
            stage("metrics.json", lambda t: t.write_text(metrics_text, encoding="utf-8"))
            for tmp, final in staged:  # metrics.json last: it is what the app checks first
                os.replace(tmp, final)
            # small per-run archive (committable): metrics + trial history of every version
            run_dir = out / "runs" / version
            run_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(run_dir / "metrics.json", metrics_text)
            if (out / "trials.csv").exists():
                atomic_write_bytes(run_dir / "trials.csv", (out / "trials.csv").read_bytes())
            R.log.log("save", "model_version", version)
            view.advance("save", "metrics.json")
            st["note"] = f"{out} (model {version})"

    atomic_write_text(
        status_path,
        json.dumps(
            {
                "state": "finished",
                "pid": os.getpid(),
                "started": started_at,
                "finished": M["run"]["finished_at"],
                "model_version": M["run"]["model_version"],
            }
        ),
    )
    R.log.log("run", "finished", M["run"]["total_seconds"])
    print_summary(view, M)
    return M


def print_summary(view: LiveView, M: dict):
    from rich.table import Table

    from rich import box

    t = Table(title="Validation leaderboard (PR-AUC)", box=box.SIMPLE_HEAD, padding=(0, 1))
    for c in ["model", "PR-AUC", "ROC-AUC", "Brier", "fit s"]:
        t.add_column(c, justify="right" if c != "model" else "left")
    for r in M["leaderboard"]:
        name = r["model"].replace("LightGBM", "LGBM").replace("Logistic regression", "LogReg").replace(" (Optuna)", "")
        t.add_row(
            ("* " if r["selected"] else "  ") + name,
            f"{r['pr_auc']:.4f}",
            f"{r['roc_auc']:.4f}",
            f"{r['brier']:.5f}",
            f"{r['fit_seconds']:.1f}",
        )
    view.print(t)
    te = M["test"]
    a = te["at_threshold"]
    view.print(
        f"[bold]Test (once):[/] PR-AUC {te['pr_auc']:.4f} (95% CI {te['pr_auc_ci95'][0]:.4f}-"
        f"{te['pr_auc_ci95'][1]:.4f}), recall {a['recall']:.3f}, precision {a['precision']:.3f}, "
        f"false alarms/10k genuine {a['false_alarms_per_10k_genuine']:.1f}, "
        f"saved {a['saved_vs_no_model']:,.0f} of {a['fraud_amount']:,.0f} at threshold {a['threshold']:.4f}"
    )
    view.print(f"Total time {M['run']['total_seconds']:.0f}s on {M['machine']['cpu_count']} CPUs")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the FraudLens model")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--trials", type=int, default=None, help="Optuna trials (default from config; 3 with --fast)")
    ap.add_argument("--fast", action="store_true", help="smoke run on a sample of cards (~2 minutes)")
    ap.add_argument("--sample-cards", type=int, default=None, help="train on a random subset of cards")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="artifacts folder")
    ap.add_argument("--no-live", action="store_true", help="plain log lines instead of the live view")
    args = ap.parse_args(argv)
    from .config import load_config

    cfg = load_config(args.config).model_dump()  # validated; a bad value stops here with a clear message
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(cfg, out, args.fast, args.trials, args.sample_cards, not args.no_live)
    except BaseException as e:  # record failures for the Training Lab, then re-raise
        atomic_write_text(
            out / "run_status.json", json.dumps({"state": "failed", "pid": os.getpid(), "error": repr(e)})
        )
        with open(out / "training_log.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "stage": "run",
                        "metric": "failed",
                        "value": repr(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
        raise


if __name__ == "__main__":
    sys.exit(main())
