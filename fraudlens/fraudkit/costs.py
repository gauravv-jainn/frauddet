"""Money-aware evaluation: cost of missed fraud plus analyst review cost."""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_at(y, p, amt, threshold: float, review_cost: float) -> dict:
    """All business metrics at one threshold (flag when p >= threshold)."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype="float64")
    amt = np.asarray(amt, dtype="float64")
    flag = p >= threshold
    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())
    tn = int((~flag & ~y).sum())
    fraud_amt = float(amt[y].sum())
    caught_amt = float(amt[flag & y].sum())
    missed_amt = fraud_amt - caught_amt
    review = review_cost * int(flag.sum())
    total_cost = missed_amt + review
    return {
        "threshold": float(threshold),
        "review_cost": float(review_cost),
        "transactions": int(len(y)),
        "flagged": int(flag.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "false_alarms_per_10k_genuine": 1e4 * fp / (fp + tn) if (fp + tn) else 0.0,
        "fraud_amount": fraud_amt,
        "caught_amount": caught_amt,
        "missed_amount": missed_amt,
        "review_cost_total": review,
        "total_cost": total_cost,
        "cost_no_model": fraud_amt,  # no model: every fraud goes through, nobody is reviewed
        "saved_vs_no_model": fraud_amt - total_cost,
    }


def cost_curve(y, p, amt, review_cost: float, max_points: int = 400) -> pd.DataFrame:
    """Total cost for every distinct threshold (exact, via sorting), down-sampled for plotting."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype="float64")
    amt = np.asarray(amt, dtype="float64")
    order = np.argsort(-p, kind="mergesort")
    ps, ys, am = p[order], y[order], amt[order]
    # candidate thresholds = distinct scores; flagging everything with p >= thr
    last_of_value = np.r_[ps[1:] != ps[:-1], True]
    n_flag = np.arange(1, len(ps) + 1)[last_of_value]
    caught = np.cumsum(np.where(ys, am, 0.0))[last_of_value]
    tp = np.cumsum(ys)[last_of_value]
    thr = ps[last_of_value]
    fraud_amt = am[ys].sum()
    n_pos = ys.sum()
    n_neg = len(ys) - n_pos
    df = pd.DataFrame(
        {
            "threshold": thr,
            "flagged": n_flag,
            "missed_amount": fraud_amt - caught,
            "review_cost_total": review_cost * n_flag,
            "recall": tp / max(n_pos, 1),
            "precision": tp / n_flag,
            "false_alarms_per_10k_genuine": 1e4 * (n_flag - tp) / max(n_neg, 1),
        }
    )
    # "flag nothing" point
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "threshold": [np.inf],
                    "flagged": [0],
                    "missed_amount": [fraud_amt],
                    "review_cost_total": [0.0],
                    "recall": [0.0],
                    "precision": [1.0],
                    "false_alarms_per_10k_genuine": [0.0],
                }
            ),
            df,
        ],
        ignore_index=True,
    )
    df["total_cost"] = df["missed_amount"] + df["review_cost_total"]
    return df


def best_threshold(y, p, amt, review_cost: float) -> tuple[float, pd.DataFrame]:
    curve = cost_curve(y, p, amt, review_cost)
    finite = curve[np.isfinite(curve["threshold"])]
    best = finite.loc[finite["total_cost"].idxmin()]
    return float(best["threshold"]), curve


def downsample_curve(curve: pd.DataFrame, max_points: int = 300) -> pd.DataFrame:
    c = curve[np.isfinite(curve["threshold"])]
    if len(c) <= max_points:
        return c.reset_index(drop=True)
    idx = np.unique(np.r_[np.linspace(0, len(c) - 1, max_points).astype(int), c["total_cost"].to_numpy().argmin()])
    return c.iloc[idx].reset_index(drop=True)
