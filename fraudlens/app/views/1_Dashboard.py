import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    ACTION_COLORS,
    ACTION_LABELS,
    business_metrics,
    decisions_vector,
    get_metrics,
    get_scored,
    money,
    money_short,
    page_setup,
    require_artifacts,
    sidebar_controls,
)
from fraudkit.decision import describe_policy

page_setup(
    "Dashboard",
    "Held-out test period (Jun-Dec 2020): every transaction scored once by the served model and decision policy.",
)
require_artifacts()
pol = sidebar_controls()
m = get_metrics()
scored_all = get_scored()

# ----------------------------------------------------------------------------- filters (drill-down)
dmin, dmax = scored_all["trans_date_trans_time"].min().date(), scored_all["trans_date_trans_time"].max().date()
with st.container(border=True):
    f1, f2, f3 = st.columns([1.3, 1.4, 1.3])
    period = f1.date_input("Period", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    cats = f2.multiselect("Categories", sorted(scored_all["category"].unique()), placeholder="All categories")
    states = f3.multiselect("States", sorted(scored_all["state"].unique()), placeholder="All states")
start, end = period if isinstance(period, tuple) and len(period) == 2 else (dmin, dmax)
mask = scored_all["trans_date_trans_time"].dt.date.between(start, end)
if cats:
    mask &= scored_all["category"].isin(cats)
if states:
    mask &= scored_all["state"].isin(states)
scored = scored_all[mask]
filtered = (start, end) != (dmin, dmax) or bool(cats) or bool(states)
if scored.empty:
    st.info("No transactions match these filters.")
    st.stop()

bm = business_metrics(scored, pol)
dec = decisions_vector(scored, pol)
flag = pd.Series((dec == "review") | (dec == "block"), index=scored.index)

c = st.columns(5)
c[0].metric("Transactions scored", f"{bm['transactions']:,}")
c[1].metric("Flagged", f"{bm['flagged']:,}", help="Sent to an analyst or blocked. " + describe_policy(pol))
c[2].metric(
    "Step-up",
    f"{bm['step_up']:,}",
    help="Risky but below the review bar, so the cardholder verifies instead. Only the expected-loss policy "
    "produces step-ups; with a flat threshold every risky transaction is reviewed.",
)
c[3].metric("Recall", f"{bm['recall']:.1%}", help="Share of fraud transactions that were flagged")
c[4].metric("Precision", f"{bm['precision']:.1%}", help="Share of flagged transactions that are fraud")

c = st.columns(5)
c[0].metric(
    "Fraud caught",
    money_short(bm["caught_amount"]),
    help=f"{money(bm['caught_amount'], 2)} of {money(bm['fraud_amount'], 2)} fraud in the period",
)
c[1].metric("Fraud missed", money_short(bm["missed_amount"]), help=money(bm["missed_amount"], 2))
c[2].metric(
    "Review cost",
    money_short(bm["review_cost_total"]),
    help=f"{bm['flagged']:,} reviews x {money(pol['review_cost'], 2)} = {money(bm['review_cost_total'], 2)}",
)
c[3].metric(
    "Net saved",
    money_short(bm["saved_vs_no_model"]),
    help=f"Saved vs no model: {money(bm['saved_vs_no_model'], 2)} = fraud amount - (missed fraud + review cost)",
)
c[4].metric(
    "False alarms /10k",
    f"{bm['false_alarms_per_10k_genuine']:.1f}",
    help="Genuine transactions flagged, per 10,000 genuine transactions",
)

# ----------------------------------------------------------------------------- trend + decision mix
left, right = st.columns([2.2, 1])
with left, st.container(border=True):
    day = scored["trans_date_trans_time"].dt.floor("D")
    daily = (
        pd.DataFrame(
            {
                "day": day,
                "flagged": flag.astype(int),
                "fraud_caught": scored["amt"].where(flag & (scored["is_fraud"] == 1), 0.0),
            }
        )
        .groupby("day")
        .agg(transactions=("flagged", "size"), flagged=("flagged", "sum"), caught=("fraud_caught", "sum"))
    )
    fig = go.Figure()
    fig.add_bar(x=daily.index, y=daily["flagged"], name="flagged", marker_color="#93C5FD")
    fig.add_scatter(
        x=daily.index, y=daily["caught"], name="fraud caught (€)", yaxis="y2", line=dict(color="#B91C1C", width=2)
    )
    fig.update_layout(
        title="Daily alerts and fraud caught",
        height=320,
        yaxis=dict(title="flagged transactions", rangemode="tozero"),
        yaxis2=dict(title="€ caught", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
        bargap=0.1,
    )
    st.plotly_chart(fig, width="stretch")
with right, st.container(border=True):
    mix = pd.Series(dec).value_counts().reindex(list(ACTION_LABELS), fill_value=0)
    share = mix / max(mix.sum(), 1)
    fig = go.Figure(
        go.Bar(
            y=[ACTION_LABELS[a] for a in mix.index][::-1],
            x=mix.values[::-1],
            orientation="h",
            marker_color=[ACTION_COLORS[a] for a in mix.index][::-1],
            text=[f"{n:,} ({sh:.2%})" for n, sh in zip(mix.values[::-1], share.values[::-1], strict=True)],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Decision mix",
        height=320,
        xaxis=dict(type="log", showticklabels=False, title="transactions (log scale)"),
        margin=dict(r=90),
    )
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- where alerts happen
left, right = st.columns(2)
with left, st.container(border=True):
    fl = scored[flag]
    by_cat = fl.groupby("category").agg(flagged=("p", "size"), fraud=("is_fraud", "sum")).reset_index()
    by_cat["false alarm"] = by_cat["flagged"] - by_cat["fraud"]
    by_cat = by_cat.sort_values("flagged")
    fig = px.bar(
        by_cat,
        y="category",
        x=["fraud", "false alarm"],
        barmode="stack",
        orientation="h",
        title="Flagged transactions by category",
        color_discrete_sequence=["#B91C1C", "#94A3B8"],
    )
    fig.update_layout(height=380, yaxis_title=None, xaxis_title="transactions", legend_title=None)
    st.plotly_chart(fig, width="stretch")
with right, st.container(border=True):
    t = scored["trans_date_trans_time"]
    heat = (
        pd.DataFrame({"hour": t.dt.hour, "dow": t.dt.dayofweek, "flag": flag})
        .groupby(["dow", "hour"])["flag"]
        .mean()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    fig = go.Figure(
        go.Heatmap(
            z=heat.values * 100,
            x=list(range(24)),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale="Reds",
            colorbar=dict(title="% flagged"),
            hovertemplate="%{y} %{x}:00<br>%{z:.2f}% flagged<extra></extra>",
        )
    )
    fig.update_layout(title="Alert rate by hour and weekday", height=380, xaxis=dict(dtick=2, title="hour of day"))
    st.plotly_chart(fig, width="stretch")

with st.container(border=True):
    st.markdown("**Merchants with the most fraud caught**")
    top = (
        scored.assign(flag=flag, caught=scored["amt"].where(flag & (scored["is_fraud"] == 1), 0.0))
        .groupby("merchant")
        .agg(alerts=("flag", "sum"), frauds=("is_fraud", "sum"), fraud_caught=("caught", "sum"))
        .sort_values("fraud_caught", ascending=False)
        .head(10)
        .reset_index()
    )
    st.dataframe(
        top,
        hide_index=True,
        width="stretch",
        column_config={
            "fraud_caught": st.column_config.ProgressColumn(
                "Fraud caught (€)", format="€%.0f", min_value=0.0, max_value=float(max(top["fraud_caught"].max(), 1))
            )
        },
    )

validated = st.session_state.get("pref_use_validated", True)
if validated and not filtered:
    st.caption("Validated policy and no filters: these numbers equal the test metrics in artifacts/metrics.json.")
else:
    st.caption("Custom policy or filters: numbers are recomputed live from the scored test set.")
st.caption(
    "Policy rules (large amounts, blocked cards, watched merchants...) are applied per transaction on top of "
    "this model policy; their measured effect is in the Model report."
)
