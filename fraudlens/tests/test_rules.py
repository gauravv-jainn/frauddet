"""Policy guardrails: the €12,000 / 2,500 km case the model alone scored as Low risk."""

import pandas as pd

from fraudkit.config import Settings
from fraudkit.rules import BAND_ORDER, combine_band, describe, policy_flags, rule_band, rule_hits

CFG = Settings().rules.model_dump()
RANGES = {"amt": {"min": 1.0, "max": 28949.0}, "dist_home_km": {"min": 0.0, "max": 151.0}}


def row(**kw):
    base = {
        "amt": 50.0,
        "amt_to_card_mean": 1.0,
        "dist_home_km": 40.0,
        "impossible_travel": 0,
        "n_txn_1h": 0,
        "geo_speed_kmh": 10.0,
        "amt_to_cat_mean": 1.0,
        "n_txn_24h": 2,
        "amt_sum_24h": 100.0,
        "prev_city_km": 10.0,
    }
    base.update(kw)
    return pd.Series(base)


def test_screenshot_case_is_escalated():
    r = row(amt=12_000.0, amt_to_card_mean=180.0, dist_home_km=2_500.0)  # the case from the app screenshot
    hits = {h.rule: h for h in describe(r, CFG, RANGES)}
    assert {"large_amount", "spend_spike", "far_from_home", "out_of_range"} <= set(hits)
    assert "distance from home" in hits["out_of_range"].text
    final = "Low"
    for h in hits.values():
        final = combine_band(final, h.band)
    assert final == "High"


def test_normal_transaction_triggers_nothing():
    assert describe(row(), CFG, RANGES) == []


def test_rules_never_lower_the_band():
    for band in BAND_ORDER:
        assert BAND_ORDER.index(combine_band(band, "Medium")) >= BAND_ORDER.index(band)
    assert combine_band("Critical", "High") == "Critical"
    assert combine_band("Low", "") == "Low"


def test_vectorised_rules_match_and_can_be_disabled():
    df = pd.DataFrame(
        [row(), row(amt=2500.0), row(n_txn_1h=6), row(impossible_travel=1, prev_city_km=900.0), row(dist_home_km=600.0)]
    )
    hits = rule_hits(df, CFG, RANGES)
    assert hits.any(axis=1).tolist() == [False, True, True, True, True]
    assert rule_band(hits).tolist() == ["", "High", "High", "High", "Medium"]
    flags, _ = policy_flags(df, [0.0, 0.0, 0.0, 0.0, 0.9], threshold=0.5, cfg=CFG, ranges=RANGES)
    assert flags.tolist() == [False, True, True, True, True]
    assert rule_hits(df, {**CFG, "enabled": False}, RANGES).empty
    # the model's noisy speed flag alone (short jump) is not enough for the policy rule
    assert not rule_hits(pd.DataFrame([row(impossible_travel=1, prev_city_km=60.0)]), CFG, RANGES).any(axis=None)


def test_policy_escalates_in_explain_frame(fitted_model):
    fm, raw = fitted_model
    from fraudkit.explain import explain_frame
    from fraudkit.features import build_features

    big = raw.iloc[[50]].copy()
    big["amt"] = 12_000.0
    feat = build_features(pd.concat([raw.iloc[:50], big]).reset_index(drop=True)).iloc[[-1]]
    plain = explain_frame(fm, feat)[0]
    e = explain_frame(fm, feat, policy={"rules": CFG, "ranges": RANGES})[0]
    assert e.probability == plain.probability  # rules never change the model probability
    assert e.rules and BAND_ORDER.index(e.band) >= BAND_ORDER.index("High")
    assert e.model_band == plain.band
    assert e.summary.startswith(f"{e.band} risk by policy rule")
    assert e.one_line().startswith("Policy:")
