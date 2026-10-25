import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE,
    RAISE,
    get_metrics,
    get_scored,
    get_settings,
    money,
    money_short,
    page_setup,
    policy,
    require_artifacts,
    sidebar_controls,
)
from fraudkit.decision import describe_policy, serving_policy
from fraudkit.rules import policy_impact

page_setup(
    "Model report", "Everything on this page is read from artifacts/metrics.json, written by the last training run."
)
require_artifacts()
sidebar_controls()
m = get_metrics()
te = m["test"]
at = te.get("at_policy", te["at_threshold"])  # served decision policy (falls back for older artifacts)

st.subheader("Validation leaderboard")
lb = pd.DataFrame(m["leaderboard"])
lb["selected"] = lb["selected"].map({True: "selected", False: ""})
cols = ["model", "pr_auc", "roc_auc", "brier", "fit_seconds", "selected"]
st.dataframe(
    lb[cols],
    hide_index=True,
    width="stretch",
    column_config={
        "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
        "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
        "brier": st.column_config.NumberColumn("Brier", format="%.5f"),
        "fit_seconds": st.column_config.NumberColumn("Fit (s)", format="%.1f"),
    },
)
tn = m["tuning"]
st.caption(
    f"Validation = last {m['run']['config']['data']['validation_fraction']:.0%} of the training period "
    f"({m['split']['validation']['rows']:,} rows, {m['split']['validation']['frauds']:,} frauds). Optuna: "
    f"{tn['trials']} trials, {tn['cv_folds']}-fold stratified CV on the fit period, best CV PR-AUC "
    f"{tn['best_cv_pr_auc']:.4f}."
)

st.subheader("Test results (held-out Jun-Dec 2020, evaluated once)")
c = st.columns(4)
c[0].metric("PR-AUC", f"{te['pr_auc']:.4f}", help=f"95% bootstrap CI ({te['bootstrap_resamples']} resamples)")
c[1].metric("PR-AUC 95% CI", f"{te['pr_auc_ci95'][0]:.4f} - {te['pr_auc_ci95'][1]:.4f}")
c[2].metric("ROC-AUC", f"{te['roc_auc']:.4f}")
c[3].metric(
    "Brier (calibrated)",
    f"{te['brier_after_calibration']:.5f}",
    delta=f"{te['brier_after_calibration'] - te['brier_before_calibration']:+.5f} vs uncalibrated",
    delta_color="inverse",
)
c = st.columns(4)
c[0].metric("Recall (served policy)", f"{at['recall']:.1%}")
c[1].metric("Precision (served policy)", f"{at['precision']:.1%}")
c[2].metric("False alarms / 10k genuine", f"{at['false_alarms_per_10k_genuine']:.1f}")
c[3].metric(
    "Saved vs no model",
    money_short(at["saved_vs_no_model"]),
    help=f"{money(at['saved_vs_no_model'], 2)}. Fraud {money(at['fraud_amount'])} without a model; "
    "with the model: missed "
    f"{money(at['missed_amount'])} + reviews {money(at['review_cost_total'])}",
)
served_pol = serving_policy(m, get_settings().decision.mode, critical=get_settings().bands.critical)
st.caption(
    f"Served policy (chosen on validation): {describe_policy(served_pol)}, review cost "
    f"{money(at['review_cost'], 2)} per flagged transaction. Flagged {at['flagged']:,} of {at['transactions']:,} "
    "test transactions."
)

left, right = st.columns(2)
with left:
    cal = m["calibration"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#94A3B8", dash="dot"), name="perfect"))
    for name, rows, color, dash in [
        ("test, uncalibrated", te.get("reliability_before", []), "#94A3B8", "dash"),
        ("test, calibrated", te["reliability_after"], BLUE, "solid"),
    ]:
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["mean_predicted"],
                y=d["observed_rate"],
                mode="lines+markers",
                name=name,
                line=dict(color=color, dash=dash),
                customdata=d[["n", "frauds"]],
                hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<br>%{customdata[0]:,} txns, "
                "%{customdata[1]:,} frauds<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Reliability on test ({cal['method']} calibration)",
        height=380,
        xaxis_title="mean predicted probability (bin)",
        yaxis_title="observed fraud rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Brier on validation {cal['validation_brier_before']:.5f} -> {cal['validation_brier_after']:.5f} "
        f"(fitted there); on test {te['brier_before_calibration']:.5f} -> {te['brier_after_calibration']:.5f}. "
        "Fixed probability bins; hover for the number of transactions in each bin."
    )
with right:
    cv = pd.DataFrame(m["threshold"]["validation_curve"])
    ct = pd.DataFrame(te["curve"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=cv["threshold"], y=cv["total_cost"], mode="lines", name="validation", line=dict(color=BLUE))
    )
    fig.add_trace(
        go.Scatter(x=ct["threshold"], y=ct["total_cost"], mode="lines", name="test", line=dict(color="#94A3B8"))
    )
    vat = m["threshold"]["validation_at_threshold"]
    fig.add_trace(
        go.Scatter(
            x=[m["threshold"]["value"]],
            y=[vat["total_cost"]],
            mode="markers",
            marker=dict(size=12, color=RAISE, symbol="diamond"),
            name="validated threshold",
        )
    )
    fig.update_layout(
        title="Threshold vs total cost (missed fraud + reviews)",
        height=380,
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="threshold (calibrated probability, log scale)",
        yaxis_title="total cost, EUR (log scale)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Review cost {money(m['threshold']['review_cost'], 2)}. The sidebar review cost does not move this "
        "curve; it is the training-time choice."
    )

st.subheader("Decision policy: flat threshold vs expected loss")
dec = m.get("decision")
if not dec:
    st.info("This model was trained before the expected-loss policy existed; retrain to compare policies.")
else:
    rows = []
    for name, label in (("threshold", "Flat threshold"), ("expected_loss", "Expected loss")):
        d = dec[name]
        param = f"p >= {d['value']:.4f}" if name == "threshold" else f"p x amount >= {d['k']:g} x review cost"
        for split in ("validation", "test"):
            if split in d:
                x = d[split]
                rows.append(
                    {
                        "policy": label + (" (served)" if dec["served"] == name else ""),
                        "rule": param,
                        "data": split,
                        "flagged": x["flagged"],
                        "recall": x["recall"],
                        "precision": x["precision"],
                        "missed fraud": x["missed_amount"],
                        "review cost": x["review_cost_total"],
                        "total cost": x["total_cost"],
                    }
                )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "recall": st.column_config.NumberColumn(format="%.4f"),
            "precision": st.column_config.NumberColumn(format="%.3f"),
            "missed fraud": st.column_config.NumberColumn(format="€%.0f"),
            "review cost": st.column_config.NumberColumn(format="€%.0f"),
            "total cost": st.column_config.NumberColumn(format="€%.0f"),
        },
    )
    kg = pd.DataFrame(dec["expected_loss"]["k_grid"])
    fig = go.Figure(go.Scatter(x=kg["k"], y=kg["total_cost"], mode="lines+markers", line=dict(color=BLUE)))
    fig.add_trace(
        go.Scatter(
            x=[dec["expected_loss"]["k"]],
            y=[dec["expected_loss"]["validation"]["total_cost"]],
            mode="markers",
            marker=dict(size=12, color=RAISE, symbol="diamond"),
            name="chosen k",
        )
    )
    fig.update_layout(
        title="Validation cost by k (expected-loss policy)",
        height=300,
        xaxis_title="k (review when p x amount >= k x review cost)",
        yaxis_title="missed fraud + reviews (€)",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"The served policy ({dec['served'].replace('_', ' ')}) is the one with the lower total cost on validation; "
        "the test rows are the independent check, computed once."
    )

st.subheader("Policy rules on top of the model (test period)")
ra = m.get("rules_auto")
if ra:
    st.markdown(
        f"Limits derived from the data at training time: review amount **{money(ra['review_amount'])}** "
        f"({ra['derivation']['review_amount']}), far from home **{ra['far_from_home_km']:,.0f} km** "
        f"({ra['derivation']['far_from_home_km']}), spending spike **{ra['spend_spike_ratio']:g}x** the card average "
        f"and at least **{money(ra['spend_spike_min_amount'])}** ({ra['derivation']['spend_spike']})."
    )
pol = policy(serving_policy(m, get_settings().decision.mode, critical=get_settings().bands.critical))
summary, per_rule = policy_impact(
    get_scored(), pol["decision"], pol["decision"]["review_cost"], pol["rules"], pol["ranges"]
)
st.dataframe(
    summary,
    hide_index=True,
    width="stretch",
    column_config={
        "precision": st.column_config.NumberColumn(format="%.3f"),
        "recall": st.column_config.NumberColumn(format="%.4f"),
        "false_alarms_per_10k_genuine": st.column_config.NumberColumn("false alarms /10k", format="%.1f"),
        "missed_amount": st.column_config.NumberColumn(format="%.0f"),
        "review_cost_total": st.column_config.NumberColumn(format="%.0f"),
        "total_cost": st.column_config.NumberColumn(format="%.0f"),
        "saved_vs_no_model": st.column_config.NumberColumn(format="%.0f"),
    },
)
st.dataframe(per_rule, hide_index=True, width="stretch")
st.caption(
    "Rules are business policy on top of the model: they cover transactions the simulator never produced as fraud "
    "(very large amounts, merchants thousands of km away, blocked cards). High/Critical rules send a transaction to "
    "review or block it; Medium rules only ask for step-up verification. On this simulated test set they mostly add "
    "reviews; their value is protection outside the training data."
)

st.subheader("What drives the model (global SHAP)")
imp = pd.Series(m["global_importance"]["mean_abs_shap"]).sort_values()
fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[BLUE])
fig.update_layout(
    height=520,
    xaxis_title="mean |SHAP| (log-odds)",
    yaxis_title=None,
    title=f"Mean absolute contribution, {m['global_importance']['sample']}",
)
st.plotly_chart(fig, width="stretch")

st.subheader("Data card")
d = m["data"]
st.markdown(f"""
- **Source:** {d["source"]}. Simulated with the Sparkov generator: {d["train"]["cards"]:,} synthetic US cards in the
  train file, {d["test"]["cards"]:,} in the test file.
- **Train file:** {d["train"]["rows"]:,} transactions, {d["train"]["frauds"]:,} frauds ({d["train"]["fraud_rate"]:.3%}),
  {d["train"]["start"][:10]} to {d["train"]["end"][:10]}. Split by time into fit ({m["split"]["fit"]["rows"]:,} rows)
  and validation ({m["split"]["validation"]["rows"]:,} rows, from {m["split"]["validation"]["start"][:10]}).
- **Test file:** {d["test"]["rows"]:,} transactions, {d["test"]["frauds"]:,} frauds ({d["test"]["fraud_rate"]:.3%}),
  {d["test"]["start"][:10]} to {d["test"]["end"][:10]}. Used once, after the threshold was fixed.
- **Features:** {len(m["features"])}, computed only from each card's earlier transactions (tested in
  `tests/test_features.py`). Merchant is target-encoded on the fit period only (out-of-fold).
- **Currency:** the data has no currency; amounts are shown as EUR for the demo.
""")

st.subheader("Limitations")
st.markdown("""
- **Simulated data.** Fraud patterns come from a generator, so they are cleaner than real fraud; real-world
  PR-AUC would be much lower. Distance from home carries almost no signal here because the simulator places
  merchants randomly around the cardholder.
- **No real merchants, devices, IPs or chargeback delays.** Labels are available instantly and are assumed correct.
- **Drift is not monitored.** The model is trained once; there is no retraining schedule or alerting on
  score/feature drift.
- **Calibration and threshold share the validation period.** Isotonic calibration and the threshold are both
  fitted on it; the test period is the honest check.
- **Single cost model.** One flat review cost per alert and full loss of every missed fraud amount.
""")
