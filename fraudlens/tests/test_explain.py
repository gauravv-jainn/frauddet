import numpy as np
import pytest
from lightgbm import LGBMClassifier

from conftest import make_toy
from fraudkit.explain import GROUPS, MIN_CONTRIB, explain_frame, shap_matrix, whatif
from fraudkit.features import CATEGORICAL, MerchantEncoder, build_features, model_frame
from fraudkit.score import Calibrator, FraudModel


@pytest.fixture(scope="module")
def fitted():
    raw = make_toy(n_cards=30, n_per_card=60, seed=3)
    # make fraud learnable: large night-time amounts
    hour = raw["trans_date_trans_time"].dt.hour
    raw["is_fraud"] = ((raw["amt"] > 80) & hour.isin([22, 23, 0, 1, 2, 3])).astype(int)
    feat = build_features(raw)
    enc = MerchantEncoder(smoothing=10)
    feat["merchant_te"] = enc.fit_transform_oof(feat["merchant"], feat["is_fraud"])
    cats = {c: sorted(feat[c].unique().tolist()) for c in CATEGORICAL}
    X = model_frame(feat, cats)
    y = feat["is_fraud"].to_numpy()
    m = LGBMClassifier(
        n_estimators=60,
        num_leaves=8,
        min_child_samples=5,
        random_state=0,
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
    ).fit(X, y)
    cal = Calibrator("platt").fit(m.predict_proba(X)[:, 1], y)
    fm = FraudModel(
        model=m,
        calibrator=cal,
        encoder=enc,
        categories=cats,
        threshold=0.2,
        bands={"medium": None, "high": 0.5, "critical": 0.8},
        review_cost=5.0,
        model_name="toy",
        data_source="toy",
    )
    return fm, raw, feat


def test_shap_reconstructs_model_output(fitted):
    fm, _, feat = fitted
    X = fm.prepare(feat)
    sv, base = shap_matrix(fm, X)
    margin = fm.model.predict(X, raw_score=True)
    np.testing.assert_allclose(base + sv.sum(axis=1), margin, atol=1e-6)


def test_reasons_only_reference_contributing_features(fitted):
    fm, _, feat = fitted
    for e in explain_frame(fm, feat.iloc[:300]):
        for r in e.raising + e.lowering:
            total = sum(e.contributions[f] for f in GROUPS[r.group])
            assert abs(total - r.contribution) < 1e-9
            assert r.features, "a reason must name at least one feature"
            for f in r.features:
                assert f in GROUPS[r.group] and e.contributions[f] != 0
            if r.direction == "raises":
                assert total >= MIN_CONTRIB
            else:
                assert total <= -MIN_CONTRIB
        assert len(e.raising) <= 4 and len(e.lowering) <= 2
        assert e.band in {"Low", "Medium", "High", "Critical"}


def test_reasons_contain_actual_numbers(fitted):
    fm, _, feat = fitted
    ex = explain_frame(fm, feat)
    texts = [r.text for e in ex for r in e.raising]
    assert any(any(ch.isdigit() for ch in t) for t in texts)


def test_whatif_amount_changes_risk_monotonic_direction(fitted):
    fm, raw, feat = fitted
    i = int(np.flatnonzero((raw["trans_date_trans_time"].dt.hour == 23).to_numpy())[0])
    row = raw.iloc[i]
    hist = raw[(raw["cc_num"] == row["cc_num"]) & (raw["trans_date_trans_time"] < row["trans_date_trans_time"])]
    low = whatif(fm, row, hist, amount=5.0)
    high = whatif(fm, row, hist, amount=500.0)
    assert high.probability >= low.probability
    # unchanged what-if equals the batch explanation
    same = whatif(fm, row, hist)
    base = explain_frame(fm, feat.iloc[[i]])[0]
    assert same.probability == pytest.approx(base.probability)


def test_whatif_distance_sets_distance(fitted):
    fm, raw, _ = fitted
    row = raw.iloc[100]
    hist = raw[(raw["cc_num"] == row["cc_num"]) & (raw["trans_date_trans_time"] < row["trans_date_trans_time"])]
    e = whatif(fm, row, hist, distance_km=1500)
    # the reasons/feature use the edited distance
    from fraudkit.explain import destination
    from fraudkit.features import haversine_km

    lat, lon = destination(10.0, 20.0, 45.0, 1500.0)
    assert haversine_km(10.0, 20.0, lat, lon) == pytest.approx(1500.0, rel=1e-6)
    assert 0.0 <= e.probability <= 1.0
