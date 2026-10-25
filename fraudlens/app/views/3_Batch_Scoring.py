import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from common import (
    ACTION_LABELS,
    BAND_COLORS,
    allowed,
    explain_kw,
    get_case_store,
    get_history,
    get_metrics,
    get_model,
    get_scored,
    get_settings,
    model_key,
    page_setup,
    require_artifacts,
    sidebar_controls,
    username,
)
from fraudkit.data import REQUIRED_COLUMNS, clean, validate_schema
from fraudkit.explain import explain_frame
from fraudkit.features import featurize_with_history
from fraudkit.security import mask_card, sanitize_csv_frame

page_setup(
    "Batch scoring",
    "Upload a CSV of transactions, get a risk-ranked table with a reason for every row.",
    permission="score",
)
require_artifacts()
pol = sidebar_controls()
thr = pol["threshold"]
fm = get_model()
max_rows = get_settings().app.max_upload_rows

with st.expander("Expected file format"):
    st.markdown(
        "One row per transaction, with these columns (same schema as the Kaggle file; extra columns are "
        f"ignored, `trans_num` is optional). Up to {max_rows:,} rows per file."
    )
    st.code(", ".join(REQUIRED_COLUMNS))
    sample = get_scored().sample(500, random_state=11).sort_values("trans_date_trans_time")
    sample = sample[REQUIRED_COLUMNS + ["trans_num"]]
    st.download_button(
        "Download a 500-row sample file",
        sanitize_csv_frame(sample).to_csv(index=False).encode(),
        "fraudlens_sample.csv",
        "text/csv",
    )

up = st.file_uploader("CSV file", type=["csv"])
use_sample = st.button("Or score the 500-row sample file")
if up is None and not use_sample and "batch_df" not in st.session_state:
    st.stop()

if up is not None:
    try:
        df = pd.read_csv(io.BytesIO(up.getvalue()), nrows=max_rows + 1, low_memory=False)
    except Exception as ex:  # malformed CSV: show the parser message, never a stack trace
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
if len(df) > max_rows:
    st.warning(f"File has more than {max_rows:,} rows; scoring the first {max_rows:,}.")
    df = df.head(max_rows)

df = df[[c for c in df.columns if c in REQUIRED_COLUMNS + ["trans_num", "is_fraud"]]].copy()
if "is_fraud" in df.columns:
    lab = pd.to_numeric(df["is_fraud"], errors="coerce")
    if lab.isna().any() or not lab.isin([0, 1]).all():
        st.warning("Column 'is_fraud' is not all 0/1; it was ignored (labels are optional for scoring).")
        df = df.drop(columns="is_fraud")
if "trans_num" not in df.columns:
    df["trans_num"] = [f"row_{i + 1}" for i in range(len(df))]
df["trans_num"] = df["trans_num"].astype(str)
df = clean(df).reset_index(drop=True)


@st.cache_data(show_spinner="Scoring and explaining", max_entries=4)
def score_batch(
    frame: pd.DataFrame, thr: float, model_version_key: str, top_raising: int, top_lowering: int, policy: dict
):
    """``model_version_key`` changes when the model file changes, so a retrain invalidates the cache."""
    hist = get_history()
    hist = hist[hist["cc_num"].isin(frame["cc_num"].unique()) & ~hist["trans_num"].isin(frame["trans_num"])]
    feat = featurize_with_history(frame, hist)
    ex = explain_frame(fm, feat, thr, top_raising, top_lowering, policy)
    out = frame[["trans_num", "trans_date_trans_time", "cc_num", "amt", "category", "merchant"]].copy()
    out["risk"] = [e.probability for e in ex]
    out["band"] = [e.band for e in ex]
    out["model_band"] = [e.model_band for e in ex]
    out["decision"] = [e.decision for e in ex]
    out["expected_loss"] = [e.expected_loss for e in ex]
    out["flagged"] = [e.flagged for e in ex]  # reviewed or blocked (model policy OR a High/Critical rule)
    out["policy_rules"] = [" | ".join(h.text for h in e.rules) for e in ex]
    out["top_reason"] = [e.one_line() for e in ex]
    out["reasons"] = [[f"Policy: {h.text}" for h in e.rules] + [r.text for r in e.raising] for e in ex]
    out["action"] = [e.action for e in ex]
    out["_rank"] = out["decision"].map({"block": 3, "review": 2, "step_up": 1, "approve": 0})
    return (
        out.sort_values(["_rank", "expected_loss", "risk"], ascending=False)
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


res = score_batch(df, thr, model_key(), **explain_kw())
c = st.columns(5)
c[0].metric("Rows scored", f"{len(res):,}")
c[1].metric("Flagged", f"{int(res['flagged'].sum()):,}")
c[2].metric("Amount flagged", f"€{res.loc[res['flagged'], 'amt'].sum():,.0f}")
c[3].metric("Blocked", f"{int((res['decision'] == 'block').sum()):,}", help="Critical risk or a blocked card")
c[4].metric(
    "Raised by policy",
    f"{int((res['band'] != res['model_band']).sum()):,}",
    help="Rows whose band was raised by a policy rule (configs/default.yaml -> rules)",
)

show = res.drop(columns=["reasons", "policy_rules"]).copy()
show["card"] = show["cc_num"].map(mask_card)
show["decision"] = show["decision"].map(ACTION_LABELS)
show = show[
    [
        "decision",
        "risk",
        "expected_loss",
        "band",
        "top_reason",
        "amt",
        "category",
        "trans_date_trans_time",
        "card",
        "merchant",
    ]
]
st.dataframe(
    show.style.map(lambda b: f"color: {BAND_COLORS.get(b, '#0F172A')}; font-weight: 600", subset=["band"]),
    width="stretch",
    height=460,
    hide_index=True,
    column_config={
        "risk": st.column_config.ProgressColumn("Model risk", format="%.3f", min_value=0.0, max_value=1.0),
        "expected_loss": st.column_config.NumberColumn("Expected loss (€)", format="%.2f"),
        "trans_date_trans_time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
        "amt": st.column_config.NumberColumn("Amount", format="%.2f"),
        "top_reason": st.column_config.TextColumn("Main reason", width="large"),
    },
)
download = res.assign(
    reasons=res["reasons"].map(" | ".join), model_version=get_metrics()["run"].get("model_version", "unversioned")
)
d1, d2 = st.columns(2)
d1.download_button(
    "Download results (CSV)",
    sanitize_csv_frame(download).to_csv(index=False).encode(),
    "fraudlens_scored.csv",
    "text/csv",
    type="primary",
)
n_flag = int(res["flagged"].sum())
if allowed("feed_queue") and d2.button(f"Send {n_flag} flagged rows to the case queue", disabled=n_flag == 0):
    store = get_case_store()
    added = 0
    version = get_metrics()["run"].get("model_version")
    for r in res[res["flagged"]].itertuples():
        cid = store.add_case(
            trans_num=r.trans_num,
            cc_num=r.cc_num,
            txn_time=r.trans_date_trans_time,
            amount=r.amt,
            category=r.category,
            merchant=r.merchant,
            score=r.risk,
            band=r.band,
            reasons=r.reasons,
            source="batch",
            actor=username(),
            model_version=version,
            expected_loss=r.expected_loss,
            decision=r.decision,
        )
        added += cid is not None
    st.success(f"Added {added} new case(s) to the queue ({n_flag - added} were already queued).")
