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


# used when a limit is "auto" but the model artifacts predate auto-derived limits
FALLBACK_AUTO = {
    "review_amount": 2000.0,
    "far_from_home_km": 500.0,
    "spend_spike_ratio": 20.0,
    "spend_spike_min_amount": 1500.0,
}


def resolve(cfg: dict, metrics: dict | None = None) -> dict:
    """Replace "auto" limits by the values derived at training time (metrics.json -> rules_auto)."""
    auto = (metrics or {}).get("rules_auto") or {}
    out = dict(cfg)
    for k, fallback in FALLBACK_AUTO.items():
        if out.get(k) == "auto":
            out[k] = float(auto.get(k, fallback))
    return out


def _watch_hits(feat: pd.DataFrame, watch: dict | None):
    from .security import card_hash

    cards = (watch or {}).get("cards") or set()
    merchants = (watch or {}).get("merchants") or set()
    card_hit = (
        feat["cc_num"].map(card_hash).isin(cards) if cards and "cc_num" in feat else pd.Series(False, index=feat.index)
    )
    merch_hit = (
        feat["merchant"].astype(str).isin(merchants)
        if merchants and "merchant" in feat
        else pd.Series(False, index=feat.index)
    )
    return card_hit, merch_hit


def rule_hits(feat: pd.DataFrame, cfg: dict, ranges: dict | None = None, watch: dict | None = None) -> pd.DataFrame:
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
    if watch:
        out["card_blocked"], out["merchant_watchlist"] = _watch_hits(feat, watch)
    return out


RULE_BANDS = {
    "large_amount": "High",
    "spend_spike": "High",
    "impossible_travel": "High",
    "burst": "High",
    "far_from_home": "Medium",
    "out_of_range": "Medium",
    "card_blocked": "Critical",
    "merchant_watchlist": "High",
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


def describe(row: pd.Series, cfg: dict, ranges: dict | None = None, watch: dict | None = None) -> list[RuleHit]:
    """Plain-English rule hits for one transaction (features + context columns)."""
    hits = rule_hits(row.to_frame().T, cfg, ranges, watch)
    if hits.empty:
        return []
    h = hits.iloc[0]
    out = []
    if h.get("card_blocked", False):
        out.append(
            RuleHit(
                "card_blocked",
                RULE_BANDS["card_blocked"],
                "This card is blocked: an analyst confirmed fraud on it earlier",
            )
        )
    if h.get("merchant_watchlist", False):
        out.append(
            RuleHit(
                "merchant_watchlist",
                RULE_BANDS["merchant_watchlist"],
                f"Merchant '{row.get('merchant')}' is on the fraud watchlist",
            )
        )
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


REVIEW_BANDS = ("High", "Critical")  # rule bands that send a transaction to an analyst (Medium = step-up only)


def policy_flags(
    feat: pd.DataFrame, p: np.ndarray, decision, cfg: dict, ranges: dict | None = None, watch: dict | None = None
):
    """Reviewed-or-blocked flags of the full system: model-side policy OR a High/Critical rule.

    ``decision`` is a decision-policy dict (fraudkit/decision.py) or, for back-compatibility, a plain threshold.
    """
    from .decision import policy_flags as model_flags

    is_policy = isinstance(decision, dict)
    base = model_flags(p, feat["amt"].to_numpy(), decision) if is_policy else np.asarray(p) >= decision
    hits = rule_hits(feat, cfg, ranges, watch)
    strong = [c for c in hits.columns if RULE_BANDS[c] in REVIEW_BANDS]
    rule_any = hits[strong].any(axis=1).to_numpy() if strong else np.zeros(len(feat), bool)
    return base | rule_any, hits, base


def policy_impact(scored: pd.DataFrame, decision, review_cost: float, cfg: dict, ranges: dict | None = None):
    """Model policy alone vs model policy + rules on a labelled, scored dataset (e.g. the test set).

    Returns (summary DataFrame with one row per system, per-rule DataFrame with hits / frauds caught).
    """
    from .costs import confusion_at

    y, amt = scored["is_fraud"].to_numpy(), scored["amt"].to_numpy()
    flags, hits, model_flag = policy_flags(scored, scored["p"].to_numpy(), decision, cfg, ranges)
    rows = []
    for name, f in [("Model policy alone", model_flag), ("Model policy + rules", flags)]:
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
    per_rule = []
    for col in hits.columns:
        h = hits[col].to_numpy()
        per_rule.append(
            {
                "rule": col,
                "band": RULE_BANDS[col],
                "effect": "review" if RULE_BANDS[col] in REVIEW_BANDS else "step-up",
                "hits": int(h.sum()),
                "frauds_in_hits": int(y[h].sum()),
                "extra_reviews": int((h & ~model_flag).sum()) if RULE_BANDS[col] in REVIEW_BANDS else 0,
                "extra_frauds_caught": int((h & ~model_flag & (y == 1)).sum())
                if RULE_BANDS[col] in REVIEW_BANDS
                else 0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(per_rule)
