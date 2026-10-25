"""Decision policy: turn a calibrated probability (plus policy rules) into an action.

Two policies are evaluated on the validation period and the cheaper one is served:

* ``threshold``: review when p >= t (one probability cut-off for every transaction).
* ``expected_loss``: review when p x amount >= k x review_cost. A calibrated probability times the amount
  at stake is the expected loss of approving; reviewing is worth it when that exceeds the review cost.
  ``k`` (how many review-costs of expected loss justify a review) is tuned on validation.

Actions, from most to least severe:
* ``block``: band Critical (model p >= bands.critical, or a Critical policy rule such as a blocked card)
* ``review``: the policy flags it, or a High policy rule fired
* ``step_up``: risky (band Medium or higher) but not worth an analyst; ask the cardholder to verify
* ``approve``: everything else

Flagged = review or block: those are the transactions that cost an analyst review in the money metric.
"""

from __future__ import annotations

import numpy as np

ACTIONS = ["approve", "step_up", "review", "block"]
ACTION_TEXT = {
    "approve": "Approve",
    "step_up": "Approve with step-up verification (one-time code to the cardholder)",
    "review": "Hold the transaction and send it to an analyst for review",
    "block": "Block the transaction and call the customer",
}
EL_K_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def policy_flags(p, amt, policy: dict) -> np.ndarray:
    """Vectorised: True where the transaction is reviewed or blocked by the model-side policy."""
    p = np.asarray(p, dtype="float64")
    amt = np.asarray(amt, dtype="float64")
    critical = p >= policy["critical"]
    if policy["mode"] == "expected_loss":
        return critical | (p * amt >= policy["k"] * policy["review_cost"])
    return critical | (p >= policy["threshold"])


def decide(p: float, amt: float, band: str, rule_band: str, policy: dict) -> tuple[str, float]:
    """Action for one transaction and its expected loss (p x amount)."""
    el = float(p) * float(amt)
    if band == "Critical" or rule_band == "Critical":
        return "block", el
    if bool(policy_flags([p], [amt], policy)[0]) or rule_band == "High":
        return "review", el
    if band != "Low" or rule_band == "Medium":
        return "step_up", el
    return "approve", el


def tune_expected_loss(y, p, amt, review_cost: float, critical: float):
    """Pick k on labelled data (validation). Returns (best_k, table of k -> metrics)."""
    from .costs import confusion_at

    rows = []
    for k in EL_K_GRID:
        flags = policy_flags(
            p, amt, {"mode": "expected_loss", "k": k, "review_cost": review_cost, "critical": critical}
        )
        m = confusion_at(y, flags.astype(float), amt, 0.5, review_cost)
        rows.append({"k": k, **{c: m[c] for c in ("flagged", "recall", "precision", "total_cost", "missed_amount")}})
    best = min(rows, key=lambda r: (r["total_cost"], -r["k"]))
    return best["k"], rows


def serving_policy(
    metrics: dict,
    mode: str = "auto",
    *,
    review_cost: float | None = None,
    threshold: float | None = None,
    k: float | None = None,
    critical: float = 0.7,
) -> dict:
    """The policy actually served: validated choices from training, optionally overridden (sidebar / config)."""
    d = metrics.get("decision") or {}
    served = d.get("served", "threshold")
    return {
        "mode": served if mode == "auto" else mode,
        "k": float(k if k is not None else d.get("expected_loss", {}).get("k", 1.0)),
        "threshold": float(threshold if threshold is not None else metrics["threshold"]["value"]),
        "review_cost": float(review_cost if review_cost is not None else metrics["threshold"]["review_cost"]),
        "critical": float(critical),
    }


def describe_policy(pol: dict) -> str:
    if pol["mode"] == "expected_loss":
        return (
            f"Review when probability x amount >= {pol['k']:g} x review cost "
            f"(>= €{pol['k'] * pol['review_cost']:,.2f} expected loss)"
        )
    return f"Review when probability >= {pol['threshold']:.4f}"
