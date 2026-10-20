import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (BAND_COLORS, get_case_store, get_history, get_model, get_scored, page_setup,  # noqa: E402
                    require_artifacts, sidebar_controls)
from fraudkit.data import REQUIRED_COLUMNS, clean, validate_schema  # noqa: E402
from fraudkit.explain import explain_frame  # noqa: E402
from fraudkit.features import featurize_with_history  # noqa: E402

MAX_ROWS = 20_000

page_setup("Batch scoring", "Upload a CSV of transactions, get a risk-ranked table with a reason for every row.")
require_artifacts()
thr, review_cost = sidebar_controls()
fm = get_model()

with st.expander("Expected file format"):
    st.markdown("One row per transaction, with these columns (same schema as the Kaggle file; extra columns are "
                "kept, `trans_num` is optional):")
    st.code(", ".join(REQUIRED_COLUMNS))
    sample = get_scored().sample(500, random_state=11).sort_values("trans_date_trans_time")
    sample = sample[REQUIRED_COLUMNS + ["trans_num"]]
    st.download_button("Download a 500-row sample file", sample.to_csv(index=False).encode(), "fraudlens_sample.csv",
                       "text/csv")

up = st.file_uploader("CSV file", type=["csv"])
use_sample = st.button("Or score the 500-row sample file")
if up is None and not use_sample and "batch_df" not in st.session_state:
    st.stop()

if up is not None:
    try:
        df = pd.read_csv(io.BytesIO(up.getvalue()))
    except Exception as ex:  # malformed CSV
        st.error(f"Could not read the file as CSV: {ex}")
        st.stop()
    st.session_state["batch_df"] = df
elif use_sample:
    st.session_state["batch_df"] = sample.reset_index(drop=True)
df = st.session_state["batch_df"]

problems = validate_schema(df)
if problems:
    st.error("The file cannot be scored:\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()
if len(df) > MAX_ROWS:
    st.warning(f"File has {len(df):,} rows; scoring the first {MAX_ROWS:,}.")
    df = df.head(MAX_ROWS)

df = df.copy()
if "trans_num" not in df.columns:
    df["trans_num"] = [f"row_{i + 1}" for i in range(len(df))]
df = clean(df).reset_index(drop=True)


@st.cache_data(show_spinner="Scoring and explaining", max_entries=4)
def score_batch(frame: pd.DataFrame, thr: float):
    hist = get_history()
    hist = hist[hist["cc_num"].isin(frame["cc_num"].unique()) & ~hist["trans_num"].isin(frame["trans_num"])]
    feat = featurize_with_history(frame, hist)
    ex = explain_frame(fm, feat, thr)
    out = frame[["trans_num", "trans_date_trans_time", "cc_num", "amt", "category", "merchant"]].copy()
    out["risk"] = [e.probability for e in ex]
    out["band"] = [e.band for e in ex]
    out["flagged"] = out["risk"] >= thr
    out["top_reason"] = [e.one_line() for e in ex]
    out["all_reasons"] = [" | ".join(r.text for r in e.raising) for e in ex]
    out["action"] = [e.action for e in ex]
    out["n_history"] = len(hist)
    return out.sort_values("risk", ascending=False).reset_index(drop=True)


res = score_batch(df, thr)
c = st.columns(4)
c[0].metric("Rows scored", f"{len(res):,}")
c[1].metric("Flagged", f"{int(res['flagged'].sum()):,}")
c[2].metric("Amount flagged", f"EUR {res.loc[res['flagged'], 'amt'].sum():,.0f}")
c[3].metric("Critical", f"{int((res['band'] == 'Critical').sum()):,}")

show = res.drop(columns=["n_history", "all_reasons"]).copy()
show["card"] = show["cc_num"].astype(str).str[-4:].radd("...")
show = show[["risk", "band", "top_reason", "amt", "category", "trans_date_trans_time", "card", "merchant", "action"]]
st.dataframe(
    show.style.map(lambda b: f"color: {BAND_COLORS.get(b, '#0F172A')}; font-weight: 600", subset=["band"]),
    width="stretch", height=460, hide_index=True,
    column_config={
        "risk": st.column_config.ProgressColumn("Risk", format="%.3f", min_value=0.0, max_value=1.0),
        "trans_date_trans_time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
        "amt": st.column_config.NumberColumn("Amount", format="%.2f"),
        "top_reason": st.column_config.TextColumn("Main reason", width="large"),
    },
)
d1, d2 = st.columns(2)
d1.download_button("Download results (CSV)", res.drop(columns=["n_history"]).to_csv(index=False).encode(),
                   "fraudlens_scored.csv", "text/csv", type="primary")
if d2.button(f"Send {int(res['flagged'].sum())} flagged rows to the case queue"):
    store = get_case_store()
    added = 0
    for r in res[res["flagged"]].itertuples():
        cid = store.add_case(trans_num=r.trans_num, cc_num=r.cc_num, txn_time=r.trans_date_trans_time, amount=r.amt,
                             category=r.category, merchant=r.merchant, score=r.risk, band=r.band,
                             reasons=r.all_reasons.split(" | ") if r.all_reasons else [], source="batch")
        added += cid is not None
    st.success(f"Added {added} new case(s) to the queue ({int(res['flagged'].sum()) - added} were already queued).")
