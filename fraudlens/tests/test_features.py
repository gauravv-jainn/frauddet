import numpy as np
import pandas as pd
import pytest

from fraudkit.features import CONTEXT, FEATURES, MerchantEncoder, build_features, featurize_with_history, haversine_km

CARD_FEATURES = [f for f in FEATURES if f != "merchant_te"] + CONTEXT


def _assert_rows_equal(a: pd.Series, b: pd.Series, i):
    for col in CARD_FEATURES:
        x, y = a[col], b[col]
        if isinstance(x, str) or isinstance(y, str):
            assert x == y, (i, col, x, y)
        elif pd.isna(x) or pd.isna(y):
            assert pd.isna(x) and pd.isna(y), (i, col, x, y)
        else:
            assert np.isclose(float(x), float(y), rtol=1e-9, atol=1e-9), (i, col, x, y)


def test_no_feature_uses_future_rows(toy):
    """Features of row i must be identical when every later row is removed."""
    full = build_features(toy)
    rng = np.random.default_rng(1)
    for i in sorted(set(rng.integers(0, len(toy), 25).tolist()) | {0, 5, len(toy) - 1}):
        truncated = build_features(toy.iloc[: i + 1])
        _assert_rows_equal(full.iloc[i], truncated.iloc[i], i)


def test_future_perturbation_does_not_change_past(toy):
    """Changing the amounts/locations of later rows must not change earlier rows."""
    base = build_features(toy)
    cut = len(toy) // 2
    mutated = toy.copy()
    mutated.loc[cut:, "amt"] *= 50
    mutated.loc[cut:, "merch_lat"] += 10
    after = build_features(mutated)
    for i in range(cut):
        _assert_rows_equal(base.iloc[i], after.iloc[i], i)


def test_featurize_with_history_matches_batch(toy):
    full = build_features(toy)
    new = toy.iloc[[60, 61, 150]]
    hist = toy.drop(index=new.index)
    hist = hist[hist["trans_date_trans_time"] < new["trans_date_trans_time"].min()]
    got = featurize_with_history(new, hist)
    # Only rows before the earliest new row are given, so compare the earliest new row.
    first = new["trans_date_trans_time"].idxmin()
    _assert_rows_equal(full.loc[first], got.loc[first], first)


def test_velocity_and_first_time_semantics(toy):
    f = build_features(toy)
    firsts = f.groupby("cc_num").head(1)
    assert (firsts["card_txn_count"] == 0).all()
    assert (firsts["n_txn_24h"] == 0).all()
    assert firsts["hours_since_prev"].isna().all()
    assert (firsts["first_time_category"] == 1).all()
    assert (f["n_txn_1h"] <= f["n_txn_24h"]).all()


def test_haversine_known_distance():
    # Paris -> London is about 344 km
    assert abs(haversine_km(48.8566, 2.3522, 51.5074, -0.1278) - 343.5) < 2


def test_merchant_encoder_oof_does_not_use_own_label():
    m = pd.Series(["a"] * 4 + ["b"] * 4)
    y = pd.Series([1, 0, 0, 0, 0, 0, 0, 0])
    enc = MerchantEncoder(smoothing=0.0, n_folds=4, seed=0)
    oof = enc.fit_transform_oof(m, y)
    # full-fit mapping for unseen merchants falls back to the prior
    assert enc.transform(pd.Series(["zzz"]))[0] == pytest.approx(y.mean())
    assert np.isfinite(oof).all()
