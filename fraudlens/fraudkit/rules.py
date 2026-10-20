"""Policy rules: a guardrail layer on top of the ML score.

Why it exists: a tree model can only reproduce patterns that are in its training data. In the simulated
data every fraud is at most ~€1,400 and every merchant is within ~150 km of home, so a €12,000 purchase
2,500 km away looks "normal" to the model. Real fraud systems therefore combine the model with explicit
business rules. The rules here are configurable (configs/default.yaml -> rules), can only RAISE the risk
band (never lower it), and every hit is shown to the analyst as its own reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BAND_ORDER = ["Low", "Medium", "High", "Critical"]
# Features checked against the range seen in training (out-of-distribution guard).
RANGE_FEATURES = [
    "amt",
    "dist_home_km",
    "amt_to_card_mean",
    "amt_to_cat_mean",
    "n_txn_1h",
    "n_txn_24h",
    "amt_sum_24h",
    "geo_speed_kmh",
]
RANGE_LABELS = {
    "amt": "amount",
    "dist_home_km": "distance from home",
    "amt_to_card_mean": "amount vs card average",
    "amt_to_cat_mean": "amount vs category average",
    "n_txn_1h": "transactions in the last hour",
    "n_txn_24h": "transactions in the last 24 hours",
    "amt_sum_24h": "24-hour spend",
    "geo_speed_kmh": "travel speed",
}


@dataclass(frozen=True)
class RuleHit:
    rule: str
    band: str  # minimum band this rule imposes
    text: str


def _money(x: float) -> str:
    return f"€{x:,.2f}"


def rule_hits(feat: pd.DataFrame, cfg: dict, ranges: dict | None = None) -> pd.DataFrame:
    """Vectorised: one boolean column per rule, aligned with ``feat``."""
    out = pd.DataFrame(index=feat.index)
    if not cfg.get("enabled", True):
        return out
    amt = feat["amt"].astype(float)
    out["large_amount"] = amt >= cfg["review_amount"]
    ratio = feat["amt_to_card_mean"].astype(float)
    out["spend_spike"] = (ratio >= cfg["spend_spike_ratio"]) & (amt >= cfg["spend_spike_min_amount"])
    out["far_from_home"] = feat["dist_home_km"].astype(float) >= cfg["far_from_home_km"]
    # The model's own impossible-travel flag is noisy in this data (merchants 50-300 km apart within minutes),
    # so the policy rule only fires for long jumps the training data never contains.
    jump = feat["prev_city_km"].astype(float) if "prev_city_km" in feat else pd.Series(0.0, index=feat.index)
    out["impossible_travel"] = (feat["impossible_travel"].astype(int) == 1) & (jump >= cfg["far_from_home_km"])
    out["burst"] = feat["n_txn_1h"].astype(float) >= cfg["burst_txn_1h"]
    if cfg.get("out_of_range", True) and ranges:
        oor = pd.Series(False, index=feat.index)
        for f in RANGE_FEATURES:
            if f in ranges and f in feat:
                v = feat[f].astype(float)
                oor |= (v > ranges[f]["max"]) | (v < ranges[f]["min"])
        out["out_of_range"] = oor
    return out


RULE_BANDS = {
    "large_amount": "High",
    "spend_spike": "High",
    "impossible_travel": "High",
    "burst": "High",
    "far_from_home": "Medium",
    "out_of_range": "Medium",
}


def rule_band(hits: pd.DataFrame) -> pd.Series:
    """Highest band imposed by any rule per row ('' when no rule fired)."""
    level = pd.Series(-1, index=hits.index)
    for col in hits.columns:
        level = level.where(
            ~hits[col] | (level >= BAND_ORDER.index(RULE_BANDS[col])), BAND_ORDER.index(RULE_BANDS[col])
        )
    return level.map(lambda i: BAND_ORDER[i] if i >= 0 else "")


def combine_band(model_band: str, rule_b: str) -> str:
    if not rule_b:
        return model_band
    return max(model_band, rule_b, key=BAND_ORDER.index)


def describe(row: pd.Series, cfg: dict, ranges: dict | None = None) -> list[RuleHit]:
    """Plain-English rule hits for one transaction (features + context columns)."""
    hits = rule_hits(row.to_frame().T, cfg, ranges)
    if hits.empty:
        return []
    h = hits.iloc[0]
    out = []
    amt = float(row["amt"])
    if h.get("large_amount", False):
        out.append(
            RuleHit(
                "large_amount",
                RULE_BANDS["large_amount"],
                f"Amount {_money(amt)} is at or above the review limit of {_money(cfg['review_amount'])}",
            )
        )
    if h.get("spend_spike", False):
        out.append(
            RuleHit(
                "spend_spike",
                RULE_BANDS["spend_spike"],
                f"Spending spike: {float(row['amt_to_card_mean']):.1f}x this card's average "
                f"(rule: {cfg['spend_spike_ratio']:g}x and at least {_money(cfg['spend_spike_min_amount'])})",
            )
        )
    if h.get("far_from_home", False):
        out.append(
            RuleHit(
                "far_from_home",
                RULE_BANDS["far_from_home"],
                f"Merchant is {float(row['dist_home_km']):,.0f} km from the cardholder's home "
                f"(rule: {cfg['far_from_home_km']:,.0f} km or more)",
            )
        )
    if h.get("impossible_travel", False):
        out.append(
            RuleHit(
                "impossible_travel",
                RULE_BANDS["impossible_travel"],
                f"Impossible travel: {float(row.get('prev_city_km', 0)):,.0f} km from the previous transaction "
                f"({float(row['geo_speed_kmh']):,.0f} km/h)",
            )
        )
    if h.get("burst", False):
        out.append(
            RuleHit(
                "burst",
                RULE_BANDS["burst"],
                f"Burst: {int(row['n_txn_1h'])} other transactions on this card in the last hour "
                f"(rule: {cfg['burst_txn_1h']} or more)",
            )
        )
    if h.get("out_of_range", False):
        names = []
        for f in RANGE_FEATURES:
            if f in (ranges or {}) and f in row and pd.notna(row[f]):
                v = float(row[f])
                if v > ranges[f]["max"] or v < ranges[f]["min"]:
                    names.append(
                        f"{RANGE_LABELS[f]} ({v:,.1f}; training range {ranges[f]['min']:,.1f}-{ranges[f]['max']:,.1f})"
                    )
        out.append(
            RuleHit(
                "out_of_range",
                RULE_BANDS["out_of_range"],
                "Outside anything the model was trained on, so the model score is unreliable here: " + "; ".join(names),
            )
        )
    return out


def policy_flags(feat: pd.DataFrame, p: np.ndarray, threshold: float, cfg: dict, ranges: dict | None = None):
    """Model flag OR rule flag, for evaluating the combined system on a whole dataset."""
    hits = rule_hits(feat, cfg, ranges)
    rule_any = hits.any(axis=1).to_numpy() if len(hits.columns) else np.zeros(len(feat), bool)
    return (np.asarray(p) >= threshold) | rule_any, hits


def policy_impact(scored: pd.DataFrame, threshold: float, review_cost: float, cfg: dict, ranges: dict | None = None):
    """Model alone vs model + policy rules on a labelled, scored dataset (e.g. the test set).

    Returns (summary DataFrame with one row per system, per-rule DataFrame with hits / frauds caught).
    """
    from .costs import confusion_at

    y, amt = scored["is_fraud"].to_numpy(), scored["amt"].to_numpy()
    flags, hits = policy_flags(scored, scored["p"].to_numpy(), threshold, cfg, ranges)
    rows = []
    for name, f in [("Model alone", scored["p"].to_numpy() >= threshold), ("Model + policy rules", flags)]:
        m = confusion_at(y, f.astype(float), amt, 0.5, review_cost)
        rows.append(
            {
                "system": name,
                **{
                    k: m[k]
                    for k in (
                        "flagged",
                        "tp",
                        "fp",
                        "fn",
                        "precision",
                        "recall",
                        "false_alarms_per_10k_genuine",
                        "missed_amount",
                        "review_cost_total",
                        "total_cost",
                        "saved_vs_no_model",
                    )
                },
            }
        )
    model_flag = scored["p"].to_numpy() >= threshold
    per_rule = []
    for col in hits.columns:
        h = hits[col].to_numpy()
        per_rule.append(
            {
                "rule": col,
                "band": RULE_BANDS[col],
                "hits": int(h.sum()),
                "frauds_in_hits": int(y[h].sum()),
                "extra_flags": int((h & ~model_flag).sum()),
                "extra_frauds_caught": int((h & ~model_flag & (y == 1)).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(per_rule)
