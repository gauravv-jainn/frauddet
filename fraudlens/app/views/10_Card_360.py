import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    ACTION_COLORS,
    ACTION_LABELS,
    allowed,
    decisions_vector,
    get_case_store,
    get_history,
    get_scored,
    get_watchlist,
    money,
    money_short,
    page_setup,
    require_artifacts,
    sidebar_controls,
    username,
)
from fraudkit.security import card_hash, esc, mask_card

page_setup("Card 360", "Everything about one card: profile, spending pattern, locations, risk over time and its cases.")
require_artifacts()
pol = sidebar_controls()
hist, scored, watch, store = get_history(), get_scored(), get_watchlist(), get_case_store()
me = username()

# ----------------------------------------------------------------------------- pick a card
risk_by_card = scored.groupby("cc_num")["p"].max().sort_values(ascending=False)
cards = risk_by_card.index.tolist()
profile_rows = hist.drop_duplicates("cc_num", keep="last").set_index("cc_num")
want = st.session_state.pop("card360_card", None)
if want is not None and want in cards:
    st.session_state["card360_select"] = want
c1, c2 = st.columns([2, 1])
cc = c1.selectbox(
    "Card",
    cards,
    key="card360_select",
    format_func=lambda c: (
        f"{mask_card(c)}  ·  {profile_rows.loc[c, 'city']}, {profile_rows.loc[c, 'state']}  ·  "
        f"max risk {risk_by_card.loc[c]:.0%}"
    ),
    help="Sorted by the card's highest risk in the test period",
)
prof = profile_rows.loc[cc]
blocked = watch.is_card_blocked(cc)
with c2:
    st.markdown(
        "<div style='margin-top:30px'>"
        + (
            "<span class='fl-chip' style='background:#B91C1C'>Blocked</span>"
            if blocked
            else "<span class='fl-chip' style='background:#15803D'>Active</span>"
        )
        + "</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------- data for this card
txns = hist[hist["cc_num"] == cc].copy()
sc = scored[scored["cc_num"] == cc]
txns = txns.merge(sc[["trans_num", "p", "dist_home_km"]], on="trans_num", how="left")
txns["period"] = np.where(txns["p"].notna(), "test (scored)", "training")
dec = pd.Series("not scored", index=txns.index)
if len(sc):
    test_mask = txns["p"].notna()
    dec[test_mask] = decisions_vector(txns[test_mask], pol)
txns["decision"] = dec
txns["label"] = np.where(txns["is_fraud"] == 1, "fraud", "genuine")
cases = store.cases()
card_cases = cases[cases["card_ref"] == card_hash(cc)] if len(cases) else cases

age = (txns["trans_date_trans_time"].max() - pd.Timestamp(prof["dob"])).days / 365.25
st.markdown(
    f"<div class='fl-muted'>{esc(prof['gender'])} &middot; age {age:.0f} &middot; {esc(prof['job'])} &middot; "
    f"home {esc(prof['city'])}, {esc(prof['state'])} (population {int(prof['city_pop']):,}) &middot; history from "
    f"{txns['trans_date_trans_time'].min():%d %b %Y}</div>",
    unsafe_allow_html=True,
)
k = st.columns(6)
k[0].metric("Transactions", f"{len(txns):,}")
k[1].metric("Total spend", money_short(txns["amt"].sum()), help=money(txns["amt"].sum(), 2))
k[2].metric("Avg amount", money(txns["amt"].mean(), 2))
k[3].metric("Frauds", int(txns["is_fraud"].sum()), help="Fraud label in the dataset")
k[4].metric(
    "Flagged", int(txns["decision"].isin(["review", "block"]).sum()), help="Reviewed or blocked in the test period"
)
k[5].metric("Cases", len(card_cases))

if allowed("decide_cases"):
    b1, b2, _ = st.columns([1, 1, 4])
    if not blocked and b1.button("Block card", icon=":material/block:"):
        watch.add_card(cc, "blocked from Card 360", actor=me)
        st.rerun()
    if blocked and b1.button("Unblock card", icon=":material/lock_open:"):
        watch.remove_card(cc, actor=me)
        st.rerun()

# ----------------------------------------------------------------------------- timeline + map
left, right = st.columns([1.5, 1])
with left, st.container(border=True):
    colors = {**{ACTION_LABELS[a]: c for a, c in ACTION_COLORS.items()}, "not scored": "#CBD5E1"}
    plot = txns.assign(decision_label=txns["decision"].map(lambda d: ACTION_LABELS.get(d, d)))
    fig = px.scatter(
        plot,
        x="trans_date_trans_time",
        y="amt",
        color="decision_label",
        symbol="label",
        symbol_map={"genuine": "circle", "fraud": "x"},
        color_discrete_map=colors,
        hover_data={"merchant": True, "category": True, "p": ":.3f", "trans_date_trans_time": "|%Y-%m-%d %H:%M"},
        log_y=True,
        title="Transactions over time (x = fraud in the dataset)",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(height=380, xaxis_title=None, yaxis_title="amount (€, log)", legend_title=None)
    st.plotly_chart(fig, width="stretch")
with right, st.container(border=True):
    geo = txns.tail(400)
    fig = go.Figure()
    fig.add_scattermap(
        lat=geo["merch_lat"],
        lon=geo["merch_long"],
        mode="markers",
        marker=dict(size=7, color=np.where(geo["is_fraud"] == 1, "#B91C1C", "#60A5FA"), opacity=0.7),
        text=geo["merchant"] + " · " + geo["amt"].map(lambda a: f"€{a:,.2f}"),
        name="merchants",
    )
    fig.add_scattermap(
        lat=[prof["lat"]],
        lon=[prof["long"]],
        mode="markers",
        marker=dict(size=16, color="#0F172A"),
        text=["home"],
        name="home",
    )
    fig.update_layout(
        title="Where the card is used",
        height=380,
        map=dict(style="open-street-map", center=dict(lat=float(prof["lat"]), lon=float(prof["long"])), zoom=6),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left, st.container(border=True):
    by_cat = txns.groupby("category").agg(spend=("amt", "sum"), n=("amt", "size"), frauds=("is_fraud", "sum"))
    by_cat = by_cat.sort_values("spend").reset_index()
    fig = px.bar(
        by_cat,
        y="category",
        x="spend",
        orientation="h",
        title="Spend by category",
        hover_data=["n", "frauds"],
        color_discrete_sequence=["#1D4ED8"],
    )
    fig.update_layout(height=340, yaxis_title=None, xaxis_title="€")
    st.plotly_chart(fig, width="stretch")
with right, st.container(border=True):
    hours = txns.assign(hour=txns["trans_date_trans_time"].dt.hour).groupby(["hour", "label"]).size()
    hours = hours.unstack(fill_value=0).reindex(range(24), fill_value=0)
    fig = go.Figure()
    for lab, col in (("genuine", "#93C5FD"), ("fraud", "#B91C1C")):
        if lab in hours:
            fig.add_bar(x=hours.index, y=hours[lab], name=lab, marker_color=col)
    fig.update_layout(title="Usual hours of use", barmode="stack", height=340, xaxis=dict(dtick=2, title="hour"))
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- recent activity + cases
st.markdown("**Recent transactions**")
recent = txns.sort_values("trans_date_trans_time", ascending=False).head(60)
recent = recent.assign(decision=recent["decision"].map(lambda d: ACTION_LABELS.get(d, d)))
st.dataframe(
    recent[["trans_date_trans_time", "amt", "category", "merchant", "city", "dist_home_km", "p", "decision", "label"]],
    hide_index=True,
    width="stretch",
    height=320,
    column_config={
        "trans_date_trans_time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
        "amt": st.column_config.NumberColumn("Amount", format="€%.2f"),
        "dist_home_km": st.column_config.NumberColumn("km from home", format="%.0f"),
        "p": st.column_config.ProgressColumn("Risk", format="%.3f", min_value=0.0, max_value=1.0),
    },
)
st.caption("Decisions shown here are the model policy only; policy rules are applied per case.")
st.markdown("**Cases for this card**")
if len(card_cases):
    st.dataframe(
        card_cases[["case_id", "txn_time", "amount", "merchant", "status", "assignee", "decision"]],
        hide_index=True,
        width="stretch",
    )
else:
    st.caption("No cases for this card.")
