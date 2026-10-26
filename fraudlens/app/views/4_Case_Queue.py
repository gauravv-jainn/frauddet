import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    ACTION_LABELS,
    action_badge,
    allowed,
    band_chip,
    card_from_ref,
    explain_kw,
    get_case_store,
    get_metrics,
    get_model,
    get_scored,
    get_settings,
    get_watchlist,
    money,
    open_card,
    page_setup,
    policy,
    require_artifacts,
    sidebar_controls,
    username,
)
from fraudkit.cases import ACTION_TO_STATUS, STATUSES
from fraudkit.explain import explain_frame
from fraudkit.rules import policy_flags
from fraudkit.security import esc, sanitize_csv_frame

page_setup("Case queue", "Flagged transactions to work: prioritised by expected loss, with SLA, comments and audit.")
require_artifacts()
pol = sidebar_controls()
thr = pol["threshold"]
fm, scored, store, watch = get_model(), get_scored(), get_case_store(), get_watchlist()
version = get_metrics()["run"].get("model_version")
me = username()
sla = get_settings().decision.sla_hours
block_on_confirm = get_settings().rules.block_confirmed_cards
stream = scored.sort_values("trans_date_trans_time", kind="mergesort").reset_index(drop=True)
full = policy(pol)


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
            reasons=[f"Policy: {h.text}" for h in e.rules] + [x.text for x in e.raising],
            source=source,
            actor=me,
            model_version=version,
            expected_loss=e.expected_loss,
            decision=e.decision,
        )
        added += cid is not None
    return added


def decide(case_ids: list[int], action: str, note: str):
    store.bulk_update(case_ids, action, note, actor=me)
    blocked = 0
    if action == "Confirm fraud" and block_on_confirm:
        for _, c in cases[cases["case_id"].isin(case_ids)].iterrows():
            if c["card_ref"]:
                blocked += watch.add_card_ref(
                    c["card_ref"], c["cc_num"], f"case #{c['case_id']} confirmed fraud by {me}", actor=me
                )
    st.toast(f"{action}: {len(case_ids)} case(s)" + (f", {blocked} card(s) blocked" if blocked else ""))


# ----------------------------------------------------------------------------- feeding the queue
if allowed("feed_queue"):
    with st.expander("Feed the queue (demo)", expanded=store.stats()["total"] == 0, icon=":material/play_circle:"):
        f1, f2, f3, f4 = st.columns([1.3, 1.1, 1, 1])
        n_stream = f1.select_slider(
            "Stream test transactions in time order", options=[1000, 2500, 5000, 10000], value=2500
        )
        if f2.button("Start demo stream", type="primary"):
            cursor = st.session_state.get("demo_cursor", 0)
            ticker, bar = st.empty(), st.progress(0.0)
            added = flagged = 0
            for start in range(cursor, min(cursor + n_stream, len(stream)), 250):
                chunk = stream.iloc[start : start + 250]
                flags, _, _ = policy_flags(
                    chunk, chunk["p"].to_numpy(), pol, full["rules"], full["ranges"], full["watch"]
                )
                hits = chunk[flags]
                flagged += len(hits)
                added += enqueue(hits, "demo")
                done = start + len(chunk) - cursor
                bar.progress(min(done / n_stream, 1.0))
                ticker.markdown(
                    f"Streaming... {done:,} transactions up to {chunk['trans_date_trans_time'].iloc[-1]:%Y-%m-%d %H:%M}"
                    f", **{flagged}** flagged, {added} new case(s)"
                )
                time.sleep(0.05)
            st.session_state["demo_cursor"] = min(cursor + n_stream, len(stream))
            st.success(f"Stream finished: {flagged} flagged out of {min(n_stream, len(stream) - cursor):,}.")
        if f3.button("Add top 25 by risk"):
            st.success(f"Added {enqueue(scored.nlargest(25, 'p'), 'queue')} case(s).")
        if allowed("clear_queue"):
            with f4.popover("Clear queue"):
                st.caption("Deletes every case, comment and case-level audit entry (the deletion itself is logged).")
                sure = st.checkbox("I understand this cannot be undone")
                if st.button("Delete all cases", type="primary", disabled=not sure):
                    store.clear(actor=me)
                    st.session_state["demo_cursor"] = 0
                    st.rerun()

# ----------------------------------------------------------------------------- stats
s = store.stats(sla, me)
c = st.columns(6)
c[0].metric("Open", s["open"] + s["escalated"], help="Open + escalated cases")
c[1].metric("Assigned to me", s["my_open"])
c[2].metric("Overdue", s["overdue"], help=f"Open or escalated for more than {sla:g} hours")
c[3].metric("Confirmed", s["confirmed_fraud"], help="Analyst confirmed fraud")
c[4].metric("Genuine", s["genuine"], help="Analyst marked genuine")
c[5].metric(
    "Agreement",
    "n/a" if s["agreement_rate"] is None else f"{s['agreement_rate']:.0%}",
    help="Confirmed fraud / (confirmed fraud + marked genuine). Every case was flagged by the system.",
)

cases = store.cases()
if cases.empty:
    st.info("The queue is empty." + (" Use 'Feed the queue' to create cases." if allowed("feed_queue") else ""))
    st.stop()
cases["age_h"] = (
    (pd.Timestamp.now(tz="UTC") - pd.to_datetime(cases["created_at"], utc=True)).dt.total_seconds() / 3600
).round(1)
active = cases["status"].isin(["open", "escalated"])
cases["overdue"] = active & (cases["age_h"] > sla)
cases["priority"] = cases["expected_loss"].fillna(cases["score"] * cases["amount"])
cases["risk_score"] = fm.risk_score(cases["score"].to_numpy())
cases["_active"] = active
cases = cases.sort_values(["_active", "overdue", "priority"], ascending=False)  # work queue first
by_id = cases.set_index("case_id")


def case_card(c) -> str:
    over = " <span class='fl-overdue'>overdue</span>" if c["overdue"] else ""
    who = esc(c["assignee"]) if c["assignee"] else "unassigned"
    return (
        f"<div class='fl-case'><b>#{c['case_id']}</b> &middot; {money(c['amount'], 2)} &middot; {esc(c['category'])}"
        f"<br><span class='meta'>score {c['risk_score']} &middot; EL {money(c['priority'], 0)} &middot; "
        f"{c['age_h']:.1f} h{over} &middot; {who}</span></div>"
    )


# ----------------------------------------------------------------------------- board / table
view_mode = st.segmented_control("View", ["Board", "Table"], default="Board", key="queue_view")
if view_mode != "Table":
    lanes = [
        ("New", cases[(cases["status"] == "open") & cases["assignee"].isna()]),
        ("In progress", cases[(cases["status"] == "open") & cases["assignee"].notna()]),
        ("Escalated", cases[cases["status"] == "escalated"]),
        (
            "Closed",
            cases[cases["status"].isin(["confirmed_fraud", "genuine"])].sort_values("updated_at", ascending=False),
        ),
    ]
    cols = st.columns(4)
    for col, (name, lane) in zip(cols, lanes, strict=True):
        body = "".join(case_card(r) for _, r in lane.head(6).iterrows()) or "<div class='fl-muted'>empty</div>"
        more = f"<div class='fl-muted'>+ {len(lane) - 6} more</div>" if len(lane) > 6 else ""
        col.markdown(
            f"<div class='fl-lane'><h4>{name} &middot; {len(lane)}</h4>{body}{more}</div>", unsafe_allow_html=True
        )
    selected_ids: list[int] = []
else:
    f1, f2 = st.columns([2, 1])
    status_filter = f1.segmented_control("Status", ["all", *STATUSES], default="open", key="queue_status")
    mine = f2.toggle("Only my cases", value=False)
    view = cases if status_filter in (None, "all") else cases[cases["status"] == status_filter]
    if mine:
        view = view[view["assignee"] == me]
    tbl = view[
        [
            "case_id",
            "score",
            "priority",
            "band",
            "decision",
            "amount",
            "category",
            "merchant",
            "status",
            "assignee",
            "age_h",
            "overdue",
        ]
    ].copy()
    tbl["decision"] = tbl["decision"].map(lambda d: ACTION_LABELS.get(d, d or ""))
    tbl.insert(0, "select", False)
    edited = st.data_editor(
        tbl,
        hide_index=True,
        width="stretch",
        height=420,
        disabled=[c for c in tbl.columns if c != "select"],
        column_config={
            "select": st.column_config.CheckboxColumn("", width="small"),
            "score": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0.0, max_value=1.0),
            "priority": st.column_config.NumberColumn("Expected loss", format="€%.0f"),
            "amount": st.column_config.NumberColumn("Amount", format="€%.2f"),
            "age_h": st.column_config.NumberColumn("Age (h)", format="%.1f"),
        },
        key="queue_editor",
    )
    selected_ids = edited.loc[edited["select"], "case_id"].astype(int).tolist()
    b1, b2, b3, b4, b5 = st.columns([1.2, 1, 1, 1, 1.4])
    b1.markdown(f"**{len(selected_ids)} selected**")
    if allowed("decide_cases") and selected_ids:
        if b2.button("Assign to me", key="bulk_assign"):
            store.bulk_assign(selected_ids, me, actor=me)
            st.rerun()
        for col, action in zip((b3, b4, b5), ACTION_TO_STATUS, strict=True):
            if col.button(action, key=f"bulk_{action}"):
                decide(selected_ids, action, "bulk action")
                st.rerun()
    export = sanitize_csv_frame(view.assign(reasons=view["reasons"].map(" | ".join)))
    st.download_button(
        "Export these cases (CSV)",
        export.to_csv(index=False).encode(),
        "fraudlens_cases.csv",
        "text/csv",
        icon=":material/download:",
    )

# ----------------------------------------------------------------------------- case detail
st.divider()
ids = cases["case_id"].tolist()
default = selected_ids[0] if selected_ids else ids[0]
cid = st.selectbox(
    "Case",
    ids,
    index=ids.index(default),
    format_func=lambda i: (
        f"#{i}  {money(by_id.loc[i, 'amount'], 2)}  {by_id.loc[i, 'category']}  "
        f"[{by_id.loc[i, 'status']}]" + ("  OVERDUE" if by_id.loc[i, "overdue"] else "")
    ),
)
case = by_id.loc[cid]
left, right = st.columns([1.25, 1])
with left:
    st.markdown(
        f"{action_badge(case['decision'] or 'review')} &nbsp; {band_chip(case['band'])} &nbsp; risk "
        f"<b>{case['score']:.2%}</b> &middot; expected loss <b>{money(case['priority'], 2)}</b> &middot; status "
        f"<b>{esc(case['status'])}</b> &middot; assignee <b>{esc(case['assignee'] or 'unassigned')}</b>",
        unsafe_allow_html=True,
    )
    over = " &middot; <span class='fl-overdue'>past SLA</span>" if case["overdue"] else ""
    st.markdown(
        f"<div class='fl-muted'>{esc(case['txn_time'])} &middot; {esc(case['merchant'])} &middot; card "
        f"{esc(case['cc_num'])} &middot; open {case['age_h']:.1f} h of {sla:g} h SLA{over} &middot; model "
        f"{esc(case.get('model_version') or 'n/a')}</div>",
        unsafe_allow_html=True,
    )
    for t in case["reasons"]:
        kind = "rule" if t.startswith("Policy:") else "raise"
        st.markdown(f"<div class='fl-card {kind}'><div class='t'>{esc(t)}</div></div>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    cc = card_from_ref(case["card_ref"])
    if cc is not None and a1.button("Open card 360", icon=":material/credit_card:", key=f"c360_{cid}"):
        open_card(cc)
    if allowed("decide_cases"):
        if a2.button("Watch merchant", icon=":material/visibility:", key=f"watch_{cid}"):
            new = watch.add_merchant(case["merchant"], f"from case #{cid}", actor=me)
            st.toast("Merchant added to the watchlist" if new else "Merchant already on the watchlist")
        if case["card_ref"] and a3.button("Block card", icon=":material/block:", key=f"block_{cid}"):
            new = watch.add_card_ref(case["card_ref"], case["cc_num"], f"blocked from case #{cid}", actor=me)
            st.toast("Card blocked: later transactions are declined" if new else "Card already blocked")
    if st.toggle("Show dataset label (demo only)", key=f"truth_{cid}"):
        lab = scored.loc[scored["trans_num"] == case["trans_num"], "is_fraud"]
        st.caption("Label: " + ("fraud" if len(lab) and int(lab.iloc[0]) else "genuine" if len(lab) else "unknown"))

with right:
    if allowed("decide_cases"):
        r1, r2 = st.columns(2)
        if case["assignee"] != me and r1.button("Assign to me", key=f"assign_{cid}", width="stretch"):
            store.assign(int(cid), me, actor=me)
            st.rerun()
        if case["assignee"] and r2.button("Unassign", key=f"unassign_{cid}", width="stretch"):
            store.assign(int(cid), None, actor=me)
            st.rerun()
        note = st.text_area("Decision note", value=case["analyst_note"] or "", key=f"note_{cid}", max_chars=2000)
        b = st.columns(3)
        for col, action in zip(b, ACTION_TO_STATUS, strict=True):
            if col.button(action, key=f"{action}_{cid}", width="stretch"):
                decide([int(cid)], action, note)
                st.rerun()
        if block_on_confirm:
            st.caption("Confirming fraud also blocks the card for later transactions (configs -> rules).")
    else:
        st.caption("Read-only: your role cannot change cases.")
    st.markdown("**Comments**")
    com = store.comments(int(cid))
    for _, r in com.iterrows():
        st.markdown(
            f"<div class='fl-card note'><div class='t'>{esc(r['text'])}</div>"
            f"<div class='w'>{esc(r['author'])} &middot; {esc(r['at'])}</div></div>",
            unsafe_allow_html=True,
        )
    if com.empty:
        st.caption("No comments yet.")
    if allowed("decide_cases"):
        with st.form(f"comment_{cid}", clear_on_submit=True):
            text = st.text_input("Add a comment", max_chars=4000)
            if st.form_submit_button("Post comment") and text.strip():
                store.add_comment(int(cid), me, text)
                st.rerun()
    with st.expander("Audit trail"):
        st.dataframe(
            store.audit(int(cid))[["at", "action", "from_status", "to_status", "note", "actor"]],
            hide_index=True,
            width="stretch",
        )

# ----------------------------------------------------------------------------- feedback stats
st.divider()
st.subheader("Analyst feedback")
decided = cases[cases["status"].isin(["confirmed_fraud", "genuine"])]
if decided.empty:
    st.caption("No decisions yet. Agreement statistics appear once analysts confirm or reject cases.")
else:
    g1, g2, g3 = st.columns(3)
    by_band = decided.groupby(["band", "status"]).size().reset_index(name="cases")
    fig = px.bar(
        by_band,
        x="band",
        y="cases",
        color="status",
        barmode="group",
        category_orders={"band": ["Low", "Medium", "High", "Critical"]},
        color_discrete_map={"confirmed_fraud": "#B91C1C", "genuine": "#15803D"},
        title="Decisions by risk band",
    )
    fig.update_layout(height=300, xaxis_title=None)
    g1.plotly_chart(fig, width="stretch")
    acts = store.audit()
    acts = acts[acts["action"].isin(ACTION_TO_STATUS)].groupby(["actor", "action"]).size().reset_index(name="n")
    fig = px.bar(
        acts,
        x="actor",
        y="n",
        color="action",
        title="Decisions per analyst",
        color_discrete_map={"Confirm fraud": "#B91C1C", "Mark genuine": "#15803D", "Escalate": "#CA8A04"},
    )
    fig.update_layout(height=300, xaxis_title=None, yaxis_title="decisions")
    g2.plotly_chart(fig, width="stretch")
    lab = decided.merge(scored[["trans_num", "is_fraud"]], on="trans_num", how="left")
    lab = lab.dropna(subset=["is_fraud"])
    if len(lab):
        correct = ((lab["status"] == "confirmed_fraud") == (lab["is_fraud"] == 1)).mean()
        conf = pd.crosstab(lab["status"], lab["is_fraud"].map({1: "label: fraud", 0: "label: genuine"}))
        g3.markdown(f"**Analyst decisions vs dataset label** (demo)  \n{correct:.0%} of {len(lab)} decisions match")
        g3.dataframe(conf, width="stretch")
    st.caption(
        "Analyst decisions are the feedback signal for the next model: confirmed cases become labelled training "
        "data. They come from the test period here, so they are not fed back into this model (that would leak the "
        "test set)."
    )
