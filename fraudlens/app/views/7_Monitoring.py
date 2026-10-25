import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE,
    STATUS_COLORS,
    get_model,
    get_reference_profile,
    get_scored,
    model_key,
    page_setup,
    require_artifacts,
    sidebar_controls,
)
from fraudkit.costs import confusion_at
from fraudkit.monitoring import PSI_MODERATE, PSI_SIGNIFICANT, drift_report

page_setup(
    "Monitoring",
    "Is the incoming data still like the training data, and is the model still performing? "
    "Computed month by month over the held-out test period.",
)
require_artifacts()
pol = sidebar_controls()
prof = get_reference_profile()
if prof is None:
    st.warning(
        "This model was trained before drift monitoring existed, so there is no reference profile. "
        "Retrain once (Training Lab) to create artifacts/reference_profile.json."
    )
    st.stop()
fm, scored = get_model(), get_scored()


@st.cache_data(show_spinner="Computing drift")
def monthly_drift(key: str) -> pd.DataFrame:
    X = fm.prepare(scored)
    months = scored["trans_date_trans_time"].dt.to_period("M").astype(str)
    out = []
    for m in sorted(months.unique()):
        mask = (months == m).to_numpy()
        r = drift_report(prof, X[mask], scores=scored.loc[mask, "p"].to_numpy())
        r["month"] = m
        r["rows"] = int(mask.sum())
        out.append(r)
    return pd.concat(out, ignore_index=True)


@st.cache_data(show_spinner=False)
def weekly_performance(key: str, pol_json: str) -> pd.DataFrame:
    import json

    from fraudkit.decision import policy_flags

    pol = json.loads(pol_json)
    rows = []
    for week, g in scored.groupby(scored["trans_date_trans_time"].dt.to_period("W").dt.start_time):
        flags = policy_flags(g["p"].to_numpy(), g["amt"].to_numpy(), pol).astype(float)
        m = confusion_at(g["is_fraud"], flags, g["amt"], 0.5, pol["review_cost"])
        rows.append(
            {
                "week": week,
                "transactions": m["transactions"],
                "flag_rate": m["flagged"] / m["transactions"],
                "fraud_rate": g["is_fraud"].mean(),
                "precision": m["precision"],
                "recall": m["recall"],
                "mean_score": g["p"].mean(),
            }
        )
    return pd.DataFrame(rows)


drift = monthly_drift(model_key())
latest = drift["month"].max()
cur = drift[drift["month"] == latest]
n_sig = int((cur["status"] == "significant").sum())
n_mod = int((cur["status"] == "moderate").sum())
score_psi = float(cur.loc[cur["feature"] == "model score", "psi"].iloc[0])

c = st.columns(4)
c[0].metric("Latest month", latest, help=f"{int(cur['rows'].iloc[0]):,} transactions")
c[1].metric("Features with significant drift", n_sig, help=f"PSI ≥ {PSI_SIGNIFICANT}")
c[2].metric("Features with moderate drift", n_mod, help=f"{PSI_MODERATE} ≤ PSI < {PSI_SIGNIFICANT}")
c[3].metric(
    "Score PSI (latest month)", f"{score_psi:.3f}", help="Shift of the model's score distribution vs validation"
)

alerts = drift[drift["status"] == "significant"].sort_values(["month", "psi"], ascending=[False, False])
if alerts.empty:
    st.success(
        "No significant drift in any month: every feature and the score stay within PSI < "
        f"{PSI_SIGNIFICANT} of the training reference."
    )
else:
    st.error(f"{len(alerts)} significant drift alert(s). Review the features below before trusting the scores.")
    st.dataframe(
        alerts[["month", "feature", "psi", "rows"]],
        hide_index=True,
        width="stretch",
        column_config={"psi": st.column_config.NumberColumn("PSI", format="%.3f")},
    )

st.subheader("Feature drift by month (PSI vs training reference)")
heat = drift.pivot(index="feature", columns="month", values="psi")
heat = heat.loc[heat.max(axis=1).sort_values(ascending=False).index]
fig = go.Figure(
    go.Heatmap(
        z=heat.values,
        x=heat.columns,
        y=heat.index,
        zmin=0,
        zmax=0.5,
        colorscale=[
            [0, "#DCFCE7"],
            [PSI_MODERATE / 0.5, "#FEF9C3"],
            [PSI_SIGNIFICANT / 0.5, "#FECACA"],
            [1, "#B91C1C"],
        ],
        text=heat.round(3).values,
        texttemplate="%{text}",
        colorbar=dict(title="PSI"),
    )
)
fig.update_layout(height=620, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, width="stretch")
st.caption(
    f"PSI < {PSI_MODERATE}: stable · {PSI_MODERATE}-{PSI_SIGNIFICANT}: moderate shift · ≥ {PSI_SIGNIFICANT}: "
    f"significant shift. Reference: {prof.get('reference', 'training data')}, "
    f"{prof.get('rows', 0):,} rows, model {prof.get('model_version', 'n/a')}."
)

st.subheader("Model performance by week")
perf = weekly_performance(model_key(), json.dumps(pol, sort_keys=True))
left, right = st.columns(2)
with left:
    fig = px.line(
        perf,
        x="week",
        y=["precision", "recall"],
        title="Precision and recall under the current policy",
        color_discrete_sequence=[BLUE, "#B91C1C"],
    )
    fig.update_layout(height=320, yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch")
with right:
    fig = px.line(
        perf,
        x="week",
        y=["flag_rate", "fraud_rate"],
        title="Flag rate vs actual fraud rate",
        color_discrete_sequence=[BLUE, "#64748B"],
    )
    fig.update_layout(height=320, yaxis_tickformat=".2%", xaxis_title=None, yaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch")
st.caption(
    "Labels are available immediately in this simulated data. In production, precision and recall arrive "
    "with chargeback delay, so watch the flag rate and drift first."
)

with st.expander("All months, all features"):
    st.dataframe(
        drift.assign(status=drift["status"]),
        hide_index=True,
        width="stretch",
        column_config={"psi": st.column_config.NumberColumn("PSI", format="%.3f")},
    )
    st.caption("Status colours: " + ", ".join(f"{k} = {v}" for k, v in STATUS_COLORS.items()))
