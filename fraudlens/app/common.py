"""Shared helpers for the FraudLens Streamlit pages: cached loaders, sidebar controls, styling."""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fraudkit.cases import CaseStore  # noqa: E402
from fraudkit.costs import confusion_at  # noqa: E402
from fraudkit.score import ARTIFACTS, FraudModel, load_metrics  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "default.yaml"

# ----------------------------------------------------------------------------- palette
BLUE = "#1D4ED8"
SLATE = "#475569"
GRID = "#E2E8F0"
RAISE = "#B91C1C"
LOWER = "#15803D"
BAND_COLORS = {"Low": "#15803D", "Medium": "#CA8A04", "High": "#EA580C", "Critical": "#B91C1C"}
BANDS = ["Low", "Medium", "High", "Critical"]

pio.templates["fraudlens"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#0F172A"),
        colorway=[BLUE, "#0EA5E9", "#64748B", "#CA8A04", "#B91C1C", "#15803D"],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        margin=dict(l=10, r=10, t=70, b=10), hoverlabel=dict(bgcolor="white"),
        title=dict(font=dict(size=15), y=0.97, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
)
pio.templates.default = "plotly_white+fraudlens"

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1240px;}
h1 {font-size: 1.75rem !important; font-weight: 650 !important;}
h2, h3 {font-weight: 600 !important;}
div[data-testid="stMetric"] {background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px;}
div[data-testid="stMetricLabel"] p {color: #475569; font-size: 0.8rem;}
div[data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 600;}
.fl-card {border: 1px solid #E2E8F0; border-left-width: 5px; border-radius: 8px; padding: 10px 14px;
          margin-bottom: 8px; background: #FFFFFF;}
.fl-card.raise {border-left-color: #B91C1C; background: #FEF2F2;}
.fl-card.lower {border-left-color: #15803D; background: #F0FDF4;}
.fl-card .t {font-size: 0.95rem; color: #0F172A;}
.fl-card .w {font-size: 0.75rem; color: #475569; margin-top: 2px;}
.fl-chip {display: inline-block; padding: 3px 12px; border-radius: 999px; color: white; font-weight: 600;
          font-size: 0.85rem;}
.fl-muted {color: #475569; font-size: 0.88rem;}
.fl-action {border: 1px solid #CBD5E1; border-radius: 8px; padding: 10px 14px; background: #F8FAFC;}
</style>
"""


def page_setup(title: str, subtitle: str | None = None):
    try:
        st.set_page_config(page_title=f"FraudLens - {title}", layout="wide")
    except Exception:
        pass  # already set by the navigation entry point
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='fl-muted'>{subtitle}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loaders
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def artifacts_ready() -> bool:
    return all((ARTIFACTS / f).exists() for f in ["model.joblib", "metrics.json", "test_scored.parquet"])


def require_artifacts():
    if not artifacts_ready():
        st.warning("No trained model found in artifacts/. Run `python -m fraudkit.train --config configs/default.yaml` "
                   "(or use the Training Lab page) and reload.")
        st.stop()


@st.cache_resource(show_spinner="Loading model")
def _model(mtime: float) -> FraudModel:
    return FraudModel.load(ARTIFACTS / "model.joblib")


def get_model() -> FraudModel:
    fm = _model(_mtime(ARTIFACTS / "model.joblib"))
    fm.bands = dict(get_config()["bands"])  # config is the source of truth for band boundaries
    return fm


@st.cache_data(show_spinner=False)
def _metrics(mtime: float) -> dict:
    return load_metrics(ARTIFACTS / "metrics.json")


def get_metrics() -> dict:
    return _metrics(_mtime(ARTIFACTS / "metrics.json"))


@st.cache_resource(show_spinner="Loading scored test set")
def _scored(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "test_scored.parquet")


def get_scored() -> pd.DataFrame:
    return _scored(_mtime(ARTIFACTS / "test_scored.parquet"))


@st.cache_resource(show_spinner="Loading transaction history")
def _history(mtime: float) -> pd.DataFrame:
    h = pd.read_parquet(ARTIFACTS / "history.parquet")
    return h.sort_values(["cc_num", "trans_date_trans_time"], kind="mergesort").reset_index(drop=True)


def get_history() -> pd.DataFrame:
    return _history(_mtime(ARTIFACTS / "history.parquet"))


def card_history(cc_num, before) -> pd.DataFrame:
    h = get_history()
    cards = h["cc_num"].to_numpy()
    lo, hi = np.searchsorted(cards, cc_num, side="left"), np.searchsorted(cards, cc_num, side="right")
    sub = h.iloc[lo:hi]
    return sub[sub["trans_date_trans_time"] < pd.Timestamp(before)]


@st.cache_data(show_spinner=False)
def get_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_case_store() -> CaseStore:
    return CaseStore(os.environ.get("FRAUDLENS_CASE_DB") or ROOT / get_config()["app"]["case_db"])


# ----------------------------------------------------------------------------- sidebar
def sidebar_controls() -> tuple[float, float]:
    """Threshold and review cost shared by all pages (kept in session_state)."""
    m = get_metrics()
    validated = float(m["threshold"]["value"])
    default_cost = float(m["threshold"]["review_cost"])
    ss = st.session_state
    ss.setdefault("use_validated", True)
    ss.setdefault("custom_threshold", round(validated, 3))
    ss.setdefault("review_cost", default_cost)
    with st.sidebar:
        st.markdown("### Decision settings")
        st.checkbox("Use validated threshold", key="use_validated",
                    help="The threshold that minimised missed-fraud + review cost on the validation period.")
        if ss["use_validated"]:
            thr = validated
            st.markdown(f"<div class='fl-muted'>Threshold: <b>{validated:.4f}</b> (chosen on validation)</div>",
                        unsafe_allow_html=True)
        else:
            thr = float(st.slider("Flag threshold (calibrated probability)", 0.001, 0.99, step=0.001,
                                  key="custom_threshold", format="%.3f"))
        st.number_input("Review cost per flagged transaction (EUR)", min_value=0.0, max_value=500.0, step=1.0,
                        key="review_cost")
        st.caption(f"Model: {m['selected_model']} | trained {m['run'].get('finished_at', '')[:10]}")
    return thr, float(ss["review_cost"])


# ----------------------------------------------------------------------------- helpers
def business_metrics(scored: pd.DataFrame, thr: float, review_cost: float) -> dict:
    return confusion_at(scored["is_fraud"].to_numpy(), scored["p"].to_numpy(), scored["amt"].to_numpy(), thr,
                        review_cost)


def money(x: float, decimals: int = 0) -> str:
    return f"EUR {x:,.{decimals}f}"


def money_short(x: float) -> str:
    """Compact money for KPI tiles (exact value goes in the tooltip)."""
    a = abs(x)
    if a >= 1e6:
        return f"€{x / 1e6:.2f}M"
    if a >= 1e4:
        return f"€{x / 1e3:.1f}k"
    return f"€{x:,.0f}"


def band_chip(band: str) -> str:
    return f"<span class='fl-chip' style='background:{BAND_COLORS[band]}'>{band}</span>"


def reason_cards(reasons, kind: str) -> str:
    if not reasons:
        return "<div class='fl-muted'>None above the reporting cut-off.</div>"
    out = []
    for r in reasons:
        out.append(f"<div class='fl-card {kind}'><div class='t'>{html.escape(r.text)}</div>"
                   f"<div class='w'>{'+' if r.contribution > 0 else ''}{r.contribution:.2f} log-odds</div></div>")
    return "".join(out)


def gauge(p: float, threshold: float, bounds: dict) -> go.Figure:
    steps = [
        dict(range=[0, bounds["medium"] * 100], color="#DCFCE7"),
        dict(range=[bounds["medium"] * 100, bounds["high"] * 100], color="#FEF9C3"),
        dict(range=[bounds["high"] * 100, bounds["critical"] * 100], color="#FFEDD5"),
        dict(range=[bounds["critical"] * 100, 100], color="#FEE2E2"),
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number=dict(suffix="%", valueformat=".1f"),
        gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color="#0F172A", thickness=0.25), steps=steps,
                   threshold=dict(line=dict(color=BLUE, width=3), thickness=0.85, value=threshold * 100)),
        title=dict(text="Calibrated fraud probability", font=dict(size=14)),
    ))
    fig.update_layout(height=250, margin=dict(l=40, r=40, t=50, b=10))
    return fig


def pages_list():
    return [
        ("views/1_Dashboard.py", "Dashboard"),
        ("views/2_Check_Transaction.py", "Check a transaction"),
        ("views/3_Batch_Scoring.py", "Batch scoring"),
        ("views/4_Case_Queue.py", "Case queue"),
        ("views/5_Model_Report.py", "Model report"),
        ("views/6_Training_Lab.py", "Training Lab"),
    ]
