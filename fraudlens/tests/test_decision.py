import numpy as np

from fraudkit.decision import decide, policy_flags, serving_policy, tune_expected_loss

EL = {"mode": "expected_loss", "k": 1.0, "review_cost": 5.0, "critical": 0.7}
THR = {"mode": "threshold", "threshold": 0.1, "review_cost": 5.0, "critical": 0.7}


def test_expected_loss_flags_large_amounts_not_small_ones():
    # same probability, different amount at stake
    p, amt = np.array([0.05, 0.05, 0.9]), np.array([20.0, 400.0, 3.0])
    assert policy_flags(p, amt, EL).tolist() == [False, True, True]  # 1.0 < 5 ; 20 >= 5 ; critical
    assert policy_flags(p, amt, THR).tolist() == [False, False, True]


def test_actions():
    assert decide(0.9, 5.0, "Critical", "", EL)[0] == "block"
    assert decide(0.2, 900.0, "Medium", "", EL)[0] == "review"  # expected loss 180
    assert decide(0.2, 10.0, "Medium", "", EL)[0] == "step_up"  # expected loss 2 < 5: cardholder verifies
    assert decide(0.0, 12000.0, "Low", "High", EL)[0] == "review"  # policy rule
    assert decide(0.0, 50.0, "Low", "Medium", EL)[0] == "step_up"
    assert decide(0.0, 50.0, "Low", "Critical", EL)[0] == "block"  # blocked card
    assert decide(0.001, 50.0, "Low", "", EL) == ("approve", 0.05)


def test_tuning_picks_lowest_cost_k():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    p = np.clip(y * 0.6 + rng.random(5000) * 0.2, 0, 1)
    amt = rng.lognormal(4, 1, 5000)
    k, table = tune_expected_loss(y, p, amt, 5.0, 0.7)
    assert k == min(table, key=lambda r: (r["total_cost"], -r["k"]))["k"]


def test_serving_policy_uses_validated_choice_and_overrides():
    metrics = {
        "threshold": {"value": 0.02, "review_cost": 5.0},
        "decision": {"served": "expected_loss", "expected_loss": {"k": 2.0}},
    }
    pol = serving_policy(metrics)
    assert (pol["mode"], pol["k"], pol["threshold"]) == ("expected_loss", 2.0, 0.02)
    assert serving_policy(metrics, "threshold", review_cost=10)["review_cost"] == 10.0
    assert serving_policy({"threshold": {"value": 0.1, "review_cost": 5.0}})["mode"] == "threshold"  # old artifacts
