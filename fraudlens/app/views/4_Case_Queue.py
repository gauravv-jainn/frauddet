import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    BAND_COLORS,
    allowed,
    band_chip,
    explain_kw,
    get_case_store,
    get_metrics,
    get_model,
    get_scored,
    money,
    page_setup,
    require_artifacts,
    sidebar_controls,
    username,
)
from fraudkit.cases import ACTION_TO_STATUS, STATUSES
from fraudkit.explain import explain_frame
from fraudkit.security import esc, sanitize_csv_frame

page_setup(
    "Case queue", "Flagged transactions waiting for an analyst decision, with assignment and a full audit trail."
)
require_artifacts()
thr, review_cost = sidebar_controls()
fm, scored, store = get_model(), get_scored(), get_case_store()
version = get_metrics()["run"].get("model_version")
me = username()
stream = scored.sort_values("trans_date_trans_time", kind="mergesort").reset_index(drop=True)


def enqueue(rows: pd.DataFrame, source: str) -> int:
    if rows.empty:
        return 0
    added = 0
    for e, r in zip(explain_frame(fm, rows, thr, **explain_kw()), rows.itertuples(), strict=True):
        cid = store.add_case(
            trans_num=r.trans_num,
            cc_num=r.cc_num,
            txn_time=r.trans_date_trans_time,
            amount=r.amt,
            category=r.category,
            merchant=r.merchant,
            score=e.probability,
            band=e.band,
            reasons=[x.text for x in e.raising],
            source=source,
            actor=me,
            model_version=version,
        )
        added += cid is not None
    return added


# ----------------------------------------------------------------------------- feeding the queue
if allowed("feed_queue"):
    with st.container(border=True):
        st.markdown("**Feed the queue**")
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1])
        n_stream = f1.select_slider(
            "Demo mode: stream test transactions", options=[1000, 2500, 5000, 10000], value=2500
        )
        if f2.button("Start demo stream", type="primary"):
            cursor = st.session_state.get("demo_cursor", 0)
            ticker, bar = st.empty(), st.progress(0.0)
            added = flagged = 0
            step = 250
            for start in range(cursor, min(cursor + n_stream, len(stream)), step):
                chunk = stream.iloc[start : start + step]
                hits = chunk[chunk["p"] >= thr]
                flagged += len(hits)
                added += enqueue(hits, "demo")
                done = start + len(chunk) - cursor
                bar.progress(min(done / n_stream, 1.0))
                last = chunk["trans_date_trans_time"].iloc[-1]
                ticker.markdown(
                    f"Streaming... {done:,} transactions up to {last:%Y-%m-%d %H:%M}, "
                    f"**{flagged}** flagged, {added} new case(s)"
                )
                time.sleep(0.08)
            st.session_state["demo_cursor"] = min(cursor + n_stream, len(stream))
            st.success(
                f"Stream finished: {flagged} flagged out of {min(n_stream, len(stream) - cursor):,} transactions."
            )
        if f3.button("Add top 25 by risk"):
            n = enqueue(scored.nlargest(25, "p"), "queue")
            st.success(f"Added {n} case(s).")
        if allowed("clear_queue"):
            with f4.popover("Clear queue"):
                st.caption(
                    "Deletes every case and its case-level audit trail. The deletion itself is recorded in "
                    "the admin audit log."
                )
                sure = st.checkbox("I understand this cannot be undone")
                if st.button("Delete all cases", type="primary", disabled=not sure):
                    store.clear(actor=me)
                    st.session_state["demo_cursor"] = 0
                    st.rerun()

# ----------------------------------------------------------------------------- stats
s = store.stats()
c = st.columns(6)
c[0].metric("Cases", s["total"])
c[1].metric("Open", s["open"])
c[2].metric("Confirmed", s["confirmed_fraud"], help="Analyst confirmed fraud")
c[3].metric("Genuine", s["genuine"], help="Analyst marked genuine")
c[4].metric("Escalated", s["escalated"])
c[5].metric(
    "Agreement",
    "n/a" if s["agreement_rate"] is None else f"{s['agreement_rate']:.0%}",
    help="Confirmed fraud / (confirmed fraud + marked genuine). Every case was flagged by the model.",
)

cases = store.cases()
if cases.empty:
    st.info(
        "The queue is empty."
        + (" Use demo mode or 'Add top 25 by risk' to create cases." if allowed("feed_queue") else "")
    )
    st.stop()
cases["age_h"] = (
    (pd.Timestamp.now(tz="UTC") - pd.to_datetime(cases["created_at"], utc=True)).dt.total_seconds() / 3600
).round(1)
by_id = cases.set_index("case_id")

# ----------------------------------------------------------------------------- queue table
left, right = st.columns([1.45, 1])
with left:
    f1, f2 = st.columns([2, 1])
    status_filter = f1.segmented_control("Status", ["all"] + STATUSES, default="open")
    mine = f2.toggle("Only my cases", value=False)
    view = cases if status_filter in (None, "all") else cases[cases["status"] == status_filter]
    if mine:
        view = view[view["assignee"] == me]
    tbl = view[
        ["case_id", "score", "band", "txn_time", "amount", "category", "merchant", "status", "assignee", "age_h"]
    ].copy()
    tbl["top reason"] = view["reasons"].map(lambda r: r[0] if r else "")
    st.dataframe(
        tbl.style.map(lambda b: f"color: {BAND_COLORS.get(b, '#0F172A')}; font-weight: 600", subset=["band"]),
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "score": st.column_config.ProgressColumn("Risk", format="%.3f", min_value=0.0, max_value=1.0),
            "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "age_h": st.column_config.NumberColumn("Age (h)", format="%.1f"),
            "top reason": st.column_config.TextColumn(width="large"),
        },
    )
    export = sanitize_csv_frame(view.assign(reasons=view["reasons"].map(" | ".join)))
    st.download_button(
        "Export these cases (CSV)", export.to_csv(index=False).encode(), "fraudlens_cases.csv", "text/csv"
    )

with right:
    ids = view["case_id"].tolist()
    if not ids:
        st.info("No cases match these filters.")
    else:
        cid = st.selectbox(
            "Case", ids, format_func=lambda i: f"#{i}  {money(by_id.loc[i, 'amount'], 2)}  {by_id.loc[i, 'category']}"
        )
        case = by_id.loc[cid]
        st.markdown(
            f"{band_chip(case['band'])} &nbsp; risk <b>{case['score']:.2%}</b> &middot; status "
            f"<b>{esc(case['status'])}</b> &middot; assignee <b>{esc(case['assignee'] or 'unassigned')}</b>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='fl-muted'>{esc(case['txn_time'])} &middot; {esc(case['merchant'])} &middot; card "
            f"{esc(case['cc_num'])} &middot; model {esc(case.get('model_version') or 'n/a')}</div>",
            unsafe_allow_html=True,
        )
        for t in case["reasons"]:
            st.markdown(f"<div class='fl-card raise'><div class='t'>{esc(t)}</div></div>", unsafe_allow_html=True)
        if st.toggle("Show dataset label (demo only)", key=f"truth_{cid}"):
            lab = scored.loc[scored["trans_num"] == case["trans_num"], "is_fraud"]
            st.caption("Label: " + ("fraud" if len(lab) and int(lab.iloc[0]) else "genuine" if len(lab) else "unknown"))
        if allowed("decide_cases"):
            a1, a2 = st.columns(2)
            if case["assignee"] != me and a1.button("Assign to me", key=f"assign_{cid}", width="stretch"):
                store.assign(int(cid), me, actor=me)
                st.rerun()
            if case["assignee"] and a2.button("Unassign", key=f"unassign_{cid}", width="stretch"):
                store.assign(int(cid), None, actor=me)
                st.rerun()
            note = st.text_area("Analyst note", value=case["analyst_note"] or "", key=f"note_{cid}", max_chars=2000)
            b = st.columns(3)
            for col, action in zip(b, ACTION_TO_STATUS, strict=True):
                if col.button(action, key=f"{action}_{cid}", width="stretch"):
                    store.update_status(int(cid), action, note, actor=me)
                    st.rerun()
        else:
            st.caption("Read-only: your role cannot change cases.")
        st.markdown("**Audit trail**")
        st.dataframe(
            store.audit(int(cid))[["at", "action", "from_status", "to_status", "note", "actor"]],
            hide_index=True,
            width="stretch",
            height=180,
        )

# ----------------------------------------------------------------------------- feedback stats
st.divider()
st.subheader("Analyst feedback")
decided = cases[cases["status"].isin(["confirmed_fraud", "genuine"])]
if decided.empty:
    st.caption("No decisions yet. Agreement statistics appear once analysts confirm or reject cases.")
else:
    g1, g2 = st.columns(2)
    by_band = decided.groupby(["band", "status"]).size().reset_index(name="cases")
    fig = px.bar(
        by_band,
        x="band",
        y="cases",
        color="status",
        barmode="group",
        category_orders={"band": ["Medium", "High", "Critical"]},
        color_discrete_map={"confirmed_fraud": "#B91C1C", "genuine": "#15803D"},
        title="Analyst decisions by model risk band",
    )
    fig.update_layout(height=300, xaxis_title=None)
    g1.plotly_chart(fig, width="stretch")
    by_analyst = store.audit()
    by_analyst = by_analyst[by_analyst["action"].isin(ACTION_TO_STATUS)].groupby(["actor", "action"]).size()
    by_analyst = by_analyst.reset_index(name="decisions")
    fig = px.bar(
        by_analyst,
        x="actor",
        y="decisions",
        color="action",
        title="Decisions per analyst",
        color_discrete_map={"Confirm fraud": "#B91C1C", "Mark genuine": "#15803D", "Escalate": "#CA8A04"},
    )
    fig.update_layout(height=300, xaxis_title=None)
    g2.plotly_chart(fig, width="stretch")
