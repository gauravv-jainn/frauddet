import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    BAND_COLORS,
    BANDS,
    BLUE,
    business_metrics,
    get_metrics,
    get_model,
    get_scored,
    money,
    money_short,
    page_setup,
    require_artifacts,
    sidebar_controls,
)

page_setup("Dashboard", "Every transaction in the held-out test period (Jun-Dec 2020), scored once by the final model.")
require_artifacts()
thr, review_cost = sidebar_controls()
fm, scored, m = get_model(), get_scored(), get_metrics()

bm = business_metrics(scored, thr, review_cost)
flagged = scored[scored["p"] >= thr]

c = st.columns(5)
c[0].metric("Transactions scored", f"{bm['transactions']:,}")
c[1].metric("Flagged", f"{bm['flagged']:,}", help=f"Sent for review: calibrated probability >= {thr:.4f}")
c[2].metric(
    "Flagged amount",
    money_short(flagged["amt"].sum()),
    help=f"Amount at risk in flagged transactions: {money(flagged['amt'].sum(), 2)}",
)
c[3].metric("Precision", f"{bm['precision']:.1%}", help="Share of flagged transactions that are fraud")
c[4].metric("Recall", f"{bm['recall']:.1%}", help="Share of fraud transactions that were flagged")

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
    help=f"{bm['flagged']:,} reviews x {money(review_cost, 2)} = {money(bm['review_cost_total'], 2)}",
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

st.divider()
left, right = st.columns(2)
with left:
    bands = pd.Series(fm.band(scored["p"].to_numpy(), thr)).value_counts().reindex(BANDS, fill_value=0)
    df = pd.DataFrame({"band": bands.index, "transactions": bands.values})
    fig = px.bar(
        df,
        x="band",
        y="transactions",
        color="band",
        color_discrete_map=BAND_COLORS,
        text="transactions",
        log_y=True,
        title="Risk-band distribution (log scale)",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, height=340, xaxis_title=None, yaxis_title="transactions")
    st.plotly_chart(fig, width="stretch")
with right:
    by_cat = flagged.groupby("category").agg(flagged=("p", "size"), fraud=("is_fraud", "sum")).reset_index()
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
    fig.update_layout(height=340, yaxis_title=None, xaxis_title="transactions", legend_title=None)
    st.plotly_chart(fig, width="stretch")

hours = scored.assign(hour=scored["trans_date_trans_time"].dt.hour, flag=scored["p"] >= thr)
by_hour = hours.groupby("hour").agg(flagged=("flag", "sum"), all=("flag", "size")).reset_index()
by_hour["flag_rate"] = by_hour["flagged"] / by_hour["all"]
fig = px.bar(
    by_hour,
    x="hour",
    y="flagged",
    title="Flagged transactions by hour of day",
    color_discrete_sequence=[BLUE],
    hover_data={"flag_rate": ":.2%"},
)
fig.update_layout(height=300, xaxis=dict(dtick=1), xaxis_title="hour of day", yaxis_title="flagged")
st.plotly_chart(fig, width="stretch")

if abs(thr - m["threshold"]["value"]) < 1e-12 and abs(review_cost - m["threshold"]["review_cost"]) < 1e-12:
    st.caption(
        "At the validated threshold and default review cost these numbers equal the test metrics in "
        "artifacts/metrics.json."
    )
else:
    st.caption("Custom threshold or review cost: numbers are recomputed live from the scored test set.")
