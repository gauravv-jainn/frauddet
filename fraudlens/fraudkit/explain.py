"""Per-transaction explanations: SHAP contributions -> plain-English reason codes.

Contributions are in log-odds (the LightGBM raw margin). Related features are grouped
(e.g. amt + log_amt) so a reason is stated once; a reason is only produced when its
group's total contribution points in the stated direction by at least MIN_CONTRIB.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .features import FEATURES, haversine_km
from .score import ACTIONS, FraudModel

MIN_CONTRIB = 0.05  # log-odds; smaller contributions are not worth a sentence
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

GROUPS: dict[str, list[str]] = {
    "amount": ["amt", "log_amt"],
    "card_ratio": ["amt_to_card_mean"],
    "cat_ratio": ["amt_to_cat_mean"],
    "time": ["hour", "is_night"],
    "weekday": ["day_of_week"],
    "distance": ["dist_home_km"],
    "velocity": ["n_txn_1h", "n_txn_24h", "amt_sum_24h"],
    "recency": ["hours_since_prev"],
    "travel": ["geo_speed_kmh", "impossible_travel"],
    "first_cat": ["first_time_category"],
    "category": ["category"],
    "merchant": ["merchant_te"],
    "age": ["age"],
    "city": ["log_city_pop"],
    "gender": ["gender"],
    "history": ["n_txn_30d"],
}
if sorted(sum(GROUPS.values(), [])) != sorted(FEATURES):
    raise RuntimeError("every feature must belong to exactly one explanation group")


@dataclass
class Reason:
    group: str
    text: str
    contribution: float  # summed SHAP (log-odds) of the group's features
    direction: str  # "raises" | "lowers"
    features: list[str] = field(default_factory=list)


@dataclass
class Explanation:
    probability: float  # calibrated
    raw_probability: float
    band: str
    action: str
    summary: str
    raising: list[Reason]
    lowering: list[Reason]
    base_value: float
    contributions: dict  # feature -> SHAP (log-odds)

    def to_dict(self):
        return asdict(self)

    def one_line(self) -> str:
        return self.raising[0].text if self.raising else "No strong risk signals"


# --------------------------------------------------------------------------- SHAP
_EXPLAINERS: dict[int, object] = {}


def explainer_for(fm: FraudModel):
    key = id(fm.model)
    if key not in _EXPLAINERS:
        import shap

        _EXPLAINERS[key] = shap.TreeExplainer(fm.model)
    return _EXPLAINERS[key]


def shap_matrix(fm: FraudModel, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    import warnings

    ex = explainer_for(fm)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
        sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    base = ex.expected_value
    base = float(base[1] if np.ndim(base) else base)
    return np.asarray(sv, dtype="float64"), base


# --------------------------------------------------------------------------- text
def _money(x: float) -> str:
    return f"€{x:,.2f}"


def _clock(ts: pd.Timestamp) -> str:
    return ts.strftime("%I:%M %p").lstrip("0")


def _fmt_minutes(m: float) -> str:
    if m < 1:
        return "under a minute"
    if m < 90:
        return f"{m:.0f} minute{'s' if round(m) != 1 else ''}"
    h = m / 60
    if h < 48:
        return f"{h:.1f} hours"
    return f"{h / 24:.0f} days"


def reason_text(group: str, raises: bool, r: pd.Series, fm: FraudModel) -> str:
    """Plain-English sentence for one feature group, with the actual numbers of this transaction."""
    amt = float(r["amt"])
    cat = str(r["category"])
    ts = pd.Timestamp(r["trans_date_trans_time"])
    if group == "amount":
        if raises:
            if amt >= 200:
                return f"Large amount of {_money(amt)}"
            if amt < 20:
                return f"Very small amount of {_money(amt)} (small 'test' charges are a fraud pattern in the data)"
            return f"Amount of {_money(amt)} is in a range where fraud is more common"
        return f"Typical amount of {_money(amt)}" if amt < 200 else f"Amount of {_money(amt)}"
    if group == "card_ratio":
        ratio = r.get("amt_to_card_mean")
        if pd.isna(ratio):
            return "No spending history on this card yet to compare the amount against"
        mean = r.get("card_mean_amt", np.nan)
        if raises and ratio >= 1.5:
            return f"Amount is {ratio:.1f}x this card's usual spend (average {_money(mean)})"
        if not raises and ratio <= 1.5:
            return f"Amount is in line with this card's usual spend ({ratio:.1f}x the average of {_money(mean)})"
        return f"Amount is {ratio:.1f}x this card's average spend of {_money(mean)}"
    if group == "cat_ratio":
        ratio = r.get("amt_to_cat_mean")
        if pd.isna(ratio):
            return f"No earlier '{cat}' purchases on this card to compare the amount against"
        mean = r.get("cat_mean_amt", np.nan)
        if raises and ratio >= 1.5:
            return f"Amount is {ratio:.1f}x what this card usually spends on '{cat}' (average {_money(mean)})"
        if not raises and ratio <= 1.5:
            return f"Amount is typical for this card's '{cat}' purchases ({ratio:.1f}x the average of {_money(mean)})"
        return f"Amount is {ratio:.1f}x this card's average '{cat}' spend of {_money(mean)}"
    if group == "time":
        if raises:
            night = " (night-time)" if int(r["is_night"]) else ""
            return f"Transaction at {_clock(ts)}{night}"
        return f"Transaction at a usual time of day ({_clock(ts)})"
    if group == "weekday":
        return f"Transaction on a {DAYS[int(r['day_of_week'])]}" + (" (riskier day in the data)" if raises else "")
    if group == "distance":
        d = float(r["dist_home_km"])
        return (
            f"Merchant is {d:,.0f} km from the cardholder's home"
            if raises
            else f"Merchant is {d:,.0f} km from the cardholder's home (normal range)"
        )
    if group == "velocity":
        n1, n24, s24 = int(r["n_txn_1h"]), int(r["n_txn_24h"]), float(r["amt_sum_24h"])
        if raises:
            if n1 >= 1:
                return (
                    f"{n1} other transaction(s) on this card in the last hour, "
                    f"{n24} in the last 24 hours ({_money(s24)})"
                )
            return f"{n24} earlier transaction(s) on this card in the last 24 hours totalling {_money(s24)}"
        return f"Usual activity level: {n24} transaction(s) in the last 24 hours ({_money(s24)})"
    if group == "recency":
        m = r.get("minutes_since_prev")
        if pd.isna(m):
            return "First transaction seen on this card"
        if raises and float(m) < 60:
            return f"Only {_fmt_minutes(float(m))} since this card's previous transaction"
        if raises:
            return f"{_fmt_minutes(float(m))} since this card's previous transaction"
        return f"{_fmt_minutes(float(m))} since this card's previous transaction"
    if group == "travel":
        km, m, spd = r.get("prev_city_km"), r.get("minutes_since_prev"), r.get("geo_speed_kmh")
        if pd.isna(km):
            return "No previous transaction location to compare with"
        if raises and int(r["impossible_travel"]):
            return (
                f"Previous transaction was {_fmt_minutes(float(m))} earlier, {km:,.0f} km away "
                f"(impossible travel, {spd:,.0f} km/h)"
            )
        if raises:
            return (
                f"Card moved {km:,.0f} km since its previous transaction "
                f"{_fmt_minutes(float(m))} earlier ({spd:,.0f} km/h)"
            )
        return f"Location is consistent with the previous transaction ({km:,.0f} km in {_fmt_minutes(float(m))})"
    if group == "first_cat":
        if int(r["first_time_category"]):
            return (
                f"First time this card is used for '{cat}'"
                if raises
                else f"First '{cat}' purchase, but this is not unusual here"
            )
        return f"This card has bought '{cat}' before" if not raises else f"Repeat '{cat}' purchase on this card"
    if group == "category":
        return (
            f"Category '{cat}' is often targeted by fraud"
            if raises
            else f"Category '{cat}' is rarely involved in fraud"
        )
    if group == "merchant":
        te, prior = float(r["merchant_te"]), fm.encoder.prior_
        m = str(r["merchant"])
        if raises:
            return f"Merchant '{m}' has an above-average fraud history ({te:.2%} vs {prior:.2%} overall)"
        return f"Merchant '{m}' has a low fraud history ({te:.2%} vs {prior:.2%} overall)"
    if group == "age":
        return f"Cardholder age {float(r['age']):.0f}" + (" (higher-risk age group in the data)" if raises else "")
    if group == "city":
        pop = int(round(np.expm1(float(r["log_city_pop"]))))
        return f"Home city population {pop:,}" + (" (higher-risk in the data)" if raises else "")
    if group == "gender":
        return f"Cardholder gender ({r['gender']}): fraud rates differ by gender in the data"
    if group == "history":
        n = int(r["n_txn_30d"])
        if raises:
            return f"Little recent activity on this card ({n} transaction(s) in the previous 30 days)"
        return f"Regular use of this card ({n} transactions in the previous 30 days)"
    raise KeyError(group)


# --------------------------------------------------------------------------- main API
def build_reasons(contrib: dict, row: pd.Series, fm: FraudModel, top_raising: int = 4, top_lowering: int = 2):
    totals = {g: float(sum(contrib[f] for f in feats)) for g, feats in GROUPS.items()}
    ups = sorted([g for g, v in totals.items() if v >= MIN_CONTRIB], key=lambda g: -totals[g])[:top_raising]
    downs = sorted([g for g, v in totals.items() if v <= -MIN_CONTRIB], key=lambda g: totals[g])[:top_lowering]

    def mk(g: str, up: bool) -> Reason:
        features = [f for f in GROUPS[g] if contrib[f] != 0]
        return Reason(g, reason_text(g, up, row, fm), totals[g], "raises" if up else "lowers", features)

    return [mk(g, True) for g in ups], [mk(g, False) for g in downs]


def _lc(text: str, strip_note: bool = True) -> str:
    """Lower-case the first letter; drop a trailing '(...)' note so summaries do not nest parentheses."""
    if strip_note:
        text = re.sub(r"\s*\([^()]*\)$", "", text)
    return text[0].lower() + text[1:]


def summarize(band: str, p: float, raising: list[Reason], lowering: list[Reason] | None = None) -> str:
    lowering = lowering or []
    if band == "Low":
        if not raising:
            return f"Low risk ({p:.1%}): nothing in this transaction stands out from the card's normal behaviour."
        if lowering:
            return (
                f"Low risk ({p:.1%}). The strongest risk signal is {_lc(raising[0].text)}, but it is "
                f"outweighed by: {_lc(lowering[0].text)}."
            )
        return f"Low risk ({p:.1%}). The strongest risk signal, {_lc(raising[0].text)}, is weak on its own."
    if not raising:
        return f"{band} risk ({p:.1%}) with no single dominant signal."
    main = _lc(raising[0].text)
    more = f", plus {len(raising) - 1} more signal{'s' if len(raising) > 2 else ''}" if len(raising) > 1 else ""
    return f"{band} risk ({p:.1%}) mainly because {main}{more}."


def explain_frame(
    fm: FraudModel, feat: pd.DataFrame, threshold: float | None = None, top_raising: int = 4, top_lowering: int = 2
) -> list[Explanation]:
    """Explain every row of a feature frame (output of build_features / test_scored)."""
    X = fm.prepare(feat)
    p_raw = fm.predict_raw(X)
    p = fm.calibrator.transform(p_raw)
    bands = fm.band(p, threshold)
    sv, base = shap_matrix(fm, X)
    out = []
    rows = feat.assign(merchant_te=X["merchant_te"].to_numpy())
    for i in range(len(X)):
        contrib = dict(zip(FEATURES, sv[i].tolist(), strict=True))
        row = rows.iloc[i]
        up, down = build_reasons(contrib, row, fm, top_raising, top_lowering)
        out.append(
            Explanation(
                probability=float(p[i]),
                raw_probability=float(p_raw[i]),
                band=str(bands[i]),
                action=ACTIONS[str(bands[i])],
                summary=summarize(str(bands[i]), float(p[i]), up, down),
                raising=up,
                lowering=down,
                base_value=base,
                contributions=contrib,
            )
        )
    return out


def whatif(
    fm: FraudModel,
    raw_row: pd.Series,
    history: pd.DataFrame,
    *,
    amount: float | None = None,
    hour: int | None = None,
    distance_km: float | None = None,
    threshold: float | None = None,
    top_raising: int = 4,
    top_lowering: int = 2,
) -> Explanation:
    """Re-score one transaction with an edited amount, hour of day and/or distance from home.

    ``history`` should contain the card's transactions strictly before the ORIGINAL transaction time.
    """
    from .features import featurize_with_history

    r = raw_row.copy()
    if amount is not None:
        r["amt"] = float(amount)
    if hour is not None:
        ts = pd.Timestamp(r["trans_date_trans_time"])
        r["trans_date_trans_time"] = ts.replace(hour=int(hour))
    if distance_km is not None:
        lat0, lon0 = float(r["lat"]), float(r["long"])
        bearing = np.arctan2(float(r["merch_long"]) - lon0, float(r["merch_lat"]) - lat0)
        if not np.isfinite(bearing) or (r["merch_lat"] == lat0 and r["merch_long"] == lon0):
            bearing = 0.0
        r["merch_lat"], r["merch_long"] = destination(lat0, lon0, np.degrees(bearing), float(distance_km))
    new = pd.DataFrame([r]).reset_index(drop=True)
    new["trans_date_trans_time"] = pd.to_datetime(new["trans_date_trans_time"])
    new["dob"] = pd.to_datetime(new["dob"])
    hist = history[history["trans_date_trans_time"] < new["trans_date_trans_time"].iloc[0]]
    feat = featurize_with_history(new, hist)
    return explain_frame(fm, feat, threshold, top_raising, top_lowering)[0]


def destination(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    """Point reached from (lat, lon) after dist_km along an initial bearing (great circle)."""
    R = 6371.0088
    phi1, lam1, th = np.radians(lat), np.radians(lon), np.radians(bearing_deg)
    d = dist_km / R
    phi2 = np.arcsin(np.sin(phi1) * np.cos(d) + np.cos(phi1) * np.sin(d) * np.cos(th))
    lam2 = lam1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(phi1), np.cos(d) - np.sin(phi1) * np.sin(phi2))
    return float(np.degrees(phi2)), float((np.degrees(lam2) + 540) % 360 - 180)


def check_distance(lat, lon, mlat, mlon) -> float:
    return float(haversine_km(lat, lon, mlat, mlon))
