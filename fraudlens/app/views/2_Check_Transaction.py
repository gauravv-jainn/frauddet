import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from common import (
    allowed,
    band_chip,
    card_history,
    explain_kw,
    gauge,
    get_history,
    get_model,
    get_scored,
    money,
    page_setup,
    reason_cards,
    rule_cards,
    require_artifacts,
    sidebar_controls,
)
from fraudkit.explain import destination, explain_frame, whatif
from fraudkit.features import featurize_with_history
from fraudkit.security import esc

page_setup("Check a transaction", "Calibrated risk, the reasons behind it, and what would change the decision.")
require_artifacts()
thr, review_cost = sidebar_controls()
fm, scored = get_model(), get_scored()
bounds = fm.band_bounds(thr)


def show(e, truth: int | None = None, key: str = "main"):
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.plotly_chart(gauge(e.probability, thr, bounds), width="stretch", key=f"gauge_{key}")
    with c2:
        st.markdown(
            f"{band_chip(e.band)} &nbsp; <b>{e.probability:.2%}</b> calibrated model probability "
            f"<span class='fl-muted'>(flag threshold {thr:.2%})</span>",
            unsafe_allow_html=True,
        )
        if e.escalated:
            st.markdown(
                f"<div class='fl-muted' style='margin-top:6px'>Model alone: {band_chip(e.model_band)} &rarr; "
                f"raised to {band_chip(e.band)} by {len(e.rules)} policy rule(s)</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"<p style='margin-top:10px'>{esc(e.summary)}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='fl-action'><b>Recommended action:</b> {esc(e.action)}</div>", unsafe_allow_html=True)
        if truth is not None:
            st.markdown(
                f"<p class='fl-muted' style='margin-top:8px'>Ground truth label in the dataset: "
                f"<b>{'fraud' if truth else 'genuine'}</b></p>",
                unsafe_allow_html=True,
            )
    if e.rules:
        st.markdown("**Policy rules triggered**")
        st.markdown(rule_cards(e.rules), unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Raises the risk (model)**")
        st.markdown(reason_cards(e.raising, "raise"), unsafe_allow_html=True)
    with r2:
        st.markdown("**Lowers the risk (model)**")
        st.markdown(reason_cards(e.lowering, "lower"), unsafe_allow_html=True)


can_score = allowed("score")
modes = ["Pick a real test transaction"] + (["Enter a transaction"] if can_score else [])
mode = st.radio("Source", modes, horizontal=True, label_visibility="collapsed")

if mode == "Pick a real test transaction":
    lists = {
        "Highest risk": scored.nlargest(300, "p"),
        "Fraud caught by the model": scored[(scored["is_fraud"] == 1) & (scored["p"] >= thr)].nlargest(300, "amt"),
        "Fraud missed by the model": scored[(scored["is_fraud"] == 1) & (scored["p"] < thr)].nlargest(300, "amt"),
        "False alarms (genuine but flagged)": scored[(scored["is_fraud"] == 0) & (scored["p"] >= thr)].nlargest(
            300, "p"
        ),
        "Random genuine": scored[scored["is_fraud"] == 0].sample(300, random_state=7),
    }
    f1, f2 = st.columns([1, 2])
    which = f1.selectbox("List", list(lists))
    pool = lists[which]
    if pool.empty:
        st.info("No transactions in this list at the current threshold.")
        st.stop()
    labels = {
        i: f"{r.trans_date_trans_time:%Y-%m-%d %H:%M}  |  {money(r.amt, 2)}  |  {r.category}  |  "
        f"{r.merchant[:28]}  |  risk {r.p:.1%}"
        for i, r in pool.iterrows()
    }
    idx = f2.selectbox("Transaction", list(labels), format_func=labels.get)
    row = scored.loc[[idx]]
    e = explain_frame(fm, row, thr, **explain_kw())[0]
    r = row.iloc[0]
    st.markdown(
        f"<div class='fl-muted'>Card ending {esc(str(r.cc_num)[-4:])} &middot; {esc(r.city)}, {esc(r.state)} "
        f"&middot; {r.trans_date_trans_time:%A %d %b %Y, %H:%M} &middot; {esc(r.merchant)} ({esc(r.category)}) "
        f"&middot; "
        f"<b>{money(r.amt, 2)}</b></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    show(e, truth=int(r.is_fraud))

    st.divider()
    st.subheader("What-if")
    if not can_score:
        st.caption("Your role is read-only; what-if re-scoring needs the analyst role.")
        st.stop()
    st.markdown(
        "<div class='fl-muted'>Edit the amount, time of day or merchant distance and re-score with the same "
        "card history. Velocity and time-since-previous features are recomputed.</div>",
        unsafe_allow_html=True,
    )
    with st.form("whatif"):
        w1, w2, w3 = st.columns(3)
        amount = w1.number_input("Amount (€)", min_value=0.0, max_value=10_000_000.0, value=float(r.amt), step=10.0)
        hour = w2.slider("Hour of day", 0, 23, int(r.trans_date_trans_time.hour))
        dist = w3.slider("Merchant distance from home (km)", 0, 3000, int(round(r.dist_home_km)))
        go_btn = st.form_submit_button("Re-score", type="primary")
    if go_btn:
        hist = card_history(r.cc_num, r.trans_date_trans_time)
        raw_cols = [c for c in get_history().columns]
        w = whatif(
            fm,
            r[raw_cols],
            hist,
            amount=amount,
            hour=hour,
            distance_km=None if dist == int(round(r.dist_home_km)) else float(dist),
            threshold=thr,
            **explain_kw(),
        )
        delta = w.probability - e.probability
        st.markdown(
            f"Risk {e.probability:.2%} &rarr; <b>{w.probability:.2%}</b> "
            f"(<span style='color:{'#B91C1C' if delta > 0 else '#15803D'}'>{delta * 100:+.1f} "
            "percentage points</span>), band "
            f"{band_chip(e.band)} &rarr; {band_chip(w.band)}",
            unsafe_allow_html=True,
        )
        show(w, key="whatif")

else:
    hist_all = get_history()
    cards = scored.drop_duplicates("cc_num").set_index("cc_num")
    card_ids = sorted(cards.index.tolist())
    with st.form("manual"):
        a1, a2, a3 = st.columns(3)
        cc = a1.selectbox(
            "Card",
            card_ids,
            format_func=lambda c: f"card ...{str(c)[-4:]} ({cards.loc[c, 'city']}, {cards.loc[c, 'state']})",
        )
        d = a2.date_input("Date", value=date(2020, 12, 15), min_value=date(2020, 6, 22), max_value=date(2021, 12, 31))
        t = a3.time_input("Time", value=time(14, 30), step=60)
        b1, b2, b3 = st.columns(3)
        amount = b1.number_input("Amount (€)", min_value=0.0, max_value=10_000_000.0, value=120.0, step=5.0)
        category = b2.selectbox("Category", fm.categories["category"])
        merchants = sorted(fm.encoder.mapping_)
        merchant = b3.selectbox("Merchant", merchants, index=0)
        c1, c2 = st.columns(2)
        dist = c1.number_input("Merchant distance from cardholder's home (km)", 0.0, 5000.0, 40.0, step=5.0)
        bearing = c2.slider("Direction from home (degrees, 0 = north)", 0, 359, 90)
        submitted = st.form_submit_button("Score transaction", type="primary")
    if submitted:
        profile = hist_all[hist_all["cc_num"] == cc].iloc[-1]
        ts = pd.Timestamp.combine(d, t)
        mlat, mlon = destination(float(profile["lat"]), float(profile["long"]), float(bearing), float(dist))
        new = pd.DataFrame(
            [
                {
                    **{
                        k: profile[k]
                        for k in ["cc_num", "gender", "city", "state", "lat", "long", "city_pop", "job", "dob"]
                    },
                    "trans_date_trans_time": ts,
                    "merchant": merchant,
                    "category": category,
                    "amt": float(amount),
                    "merch_lat": mlat,
                    "merch_long": mlon,
                    "trans_num": "manual",
                    "is_fraud": 0,
                }
            ]
        )
        hist = card_history(cc, ts)
        feat = featurize_with_history(new, hist)
        e = explain_frame(fm, feat, thr, **explain_kw())[0]
        st.markdown(
            f"<div class='fl-muted'>Scored against {len(hist):,} earlier transactions of this card.</div>",
            unsafe_allow_html=True,
        )
        show(e, key="manual")
