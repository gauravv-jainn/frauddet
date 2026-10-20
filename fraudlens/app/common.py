"""Shared helpers for the FraudLens Streamlit pages: auth gate, cached loaders, sidebar controls, styling."""

from __future__ import annotations

import contextlib
import html
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fraudkit.auth import ApiKeyStore, AuthError, UserStore, can
from fraudkit.cases import CaseStore
from fraudkit.config import Settings, load_config
from fraudkit.costs import confusion_at
from fraudkit.db import Database
from fraudkit.score import ARTIFACTS, FraudModel, load_metrics, load_serving_model

# ----------------------------------------------------------------------------- palette
BLUE = "#1D4ED8"
SLATE = "#475569"
GRID = "#E2E8F0"
RAISE = "#B91C1C"
LOWER = "#15803D"
BAND_COLORS = {"Low": "#15803D", "Medium": "#CA8A04", "High": "#EA580C", "Critical": "#B91C1C"}
BANDS = ["Low", "Medium", "High", "Critical"]
STATUS_COLORS = {"stable": "#15803D", "moderate": "#CA8A04", "significant": "#B91C1C"}

pio.templates["fraudlens"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#0F172A"),
        colorway=[BLUE, "#0EA5E9", "#64748B", "#CA8A04", "#B91C1C", "#15803D"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        margin=dict(l=10, r=10, t=70, b=10),
        hoverlabel=dict(bgcolor="white"),
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
.fl-card.rule {border-left-color: #7C3AED; background: #F5F3FF;}
.fl-card .t {font-size: 0.95rem; color: #0F172A;}
.fl-card .w {font-size: 0.75rem; color: #475569; margin-top: 2px;}
.fl-chip {display: inline-block; padding: 3px 12px; border-radius: 999px; color: white; font-weight: 600;
          font-size: 0.85rem;}
.fl-muted {color: #475569; font-size: 0.88rem;}
.fl-action {border: 1px solid #CBD5E1; border-radius: 8px; padding: 10px 14px; background: #F8FAFC;}
.fl-user {font-size: 0.85rem; color: #0F172A;}
.fl-role {display: inline-block; padding: 1px 8px; border-radius: 6px; background: #E0E7FF; color: #1E3A8A;
          font-size: 0.75rem; font-weight: 600; margin-left: 4px;}
</style>
"""


# ----------------------------------------------------------------------------- config / stores
@st.cache_resource
def get_settings() -> Settings:
    return load_config()


def get_config() -> dict:
    return get_settings().model_dump()


def policy() -> dict:
    """Policy-rule settings + training ranges, passed to every explanation (see fraudkit/rules.py)."""
    from fraudkit.monitoring import training_ranges

    return {"rules": get_settings().rules.model_dump(), "ranges": training_ranges(get_reference_profile())}


def explain_kw() -> dict:
    """Number of reasons to show (config) and the policy rules."""
    e = get_settings().explain
    return {"top_raising": e.top_raising, "top_lowering": e.top_lowering, "policy": policy()}


@st.cache_resource
def _db(path: str) -> Database:
    return Database(path)


def get_db() -> Database:
    return _db(str(get_settings().db_path()))


def get_users() -> UserStore:
    s = get_settings().app
    return UserStore(get_db(), max_failed=s.max_failed_logins, lockout_minutes=s.lockout_minutes)


def get_api_keys() -> ApiKeyStore:
    return ApiKeyStore(get_db())


def get_case_store() -> CaseStore:
    return CaseStore(get_db())


# ----------------------------------------------------------------------------- auth gate
def _now() -> datetime:
    return datetime.now(UTC)


def current_user() -> dict | None:
    return st.session_state.get("auth")


def username() -> str:
    u = current_user()
    return u["username"] if u else "anonymous"


def allowed(permission: str) -> bool:
    u = current_user()
    return bool(u) and can(u["role"], permission)


def require_permission(permission: str, what: str = "this page"):
    if not allowed(permission):
        st.info(f"Your role does not have access to {what}. Ask an administrator if you need it.")
        st.stop()


def logout(reason: str | None = None):
    st.session_state.pop("auth", None)
    if reason:
        st.session_state["logout_reason"] = reason


def _start_session(user):
    st.session_state["auth"] = {
        "username": user.username,
        "role": user.role,
        "login_at": _now().isoformat(),
        "last_seen": _now().isoformat(),
        "must_change_pw": user.must_change_pw,
    }


def _login_screen():
    users = get_users()
    st.markdown(CSS, unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown("## FraudLens")
        st.markdown("<div class='fl-muted'>Explainable card-fraud scoring</div>", unsafe_allow_html=True)
        if reason := st.session_state.pop("logout_reason", None):
            st.info(reason)
        if users.count() == 0:
            st.markdown("#### First-run setup")
            st.caption(
                "No accounts exist yet. Create the first administrator (use a password manager); "
                "this account can then create the other users."
            )
            with st.form("setup"):
                u = st.text_input("Admin username", value="admin")
                p1 = st.text_input("Password", type="password")
                p2 = st.text_input("Repeat password", type="password")
                if st.form_submit_button("Create administrator", type="primary"):
                    if p1 != p2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            user = users.create(u, p1, "admin", actor="setup")
                            _start_session(user)
                            st.rerun()
                        except AuthError as e:
                            st.error(str(e))
            return
        with st.form("login"):
            st.markdown("#### Sign in")
            u = st.text_input("Username", autocomplete="username")
            p = st.text_input("Password", type="password", autocomplete="current-password")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                try:
                    _start_session(users.authenticate(u, p))
                    st.rerun()
                except AuthError as e:
                    st.error(str(e))


def change_password_form(key: str):
    users = get_users()
    with st.form(key):
        cur = st.text_input("Current password", type="password")
        p1 = st.text_input("New password", type="password")
        p2 = st.text_input("Repeat new password", type="password")
        if st.form_submit_button("Change password", type="primary"):
            u = username()
            try:
                users.authenticate(u, cur)
                if p1 != p2:
                    raise AuthError("New passwords do not match.")
                if p1 == cur:
                    raise AuthError("Choose a password different from the current one.")
                users.set_password(u, p1, actor=u, must_change=False)
                st.session_state["auth"]["must_change_pw"] = False
                st.session_state["pw_changed"] = True
                st.rerun()
            except AuthError as e:
                st.error(str(e))


def require_login():
    """Gate every page. Renders the login (or first-run setup) and stops when nobody is signed in."""
    sess = current_user()
    if sess:
        timeout = timedelta(minutes=get_settings().app.session_timeout_minutes)
        user = get_users().get(sess["username"])
        if _now() - datetime.fromisoformat(sess["last_seen"]) > timeout:
            logout("Your session expired. Please sign in again.")
        elif user is None or not user.active:
            logout("Your account is no longer active.")
        else:
            sess["role"] = user.role  # role changes by an admin apply on the next interaction
            sess["last_seen"] = _now().isoformat()
            sess["must_change_pw"] = user.must_change_pw
    if not current_user():
        _login_screen()
        st.stop()
    if current_user().get("must_change_pw"):
        st.markdown(CSS, unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 1.3, 1])
        with mid:
            st.markdown("## Set a new password")
            st.caption("An administrator set a temporary password for this account. Choose your own to continue.")
            change_password_form("forced_pw")
        st.stop()


def sidebar_user():
    u = current_user()
    if not u:
        return
    with st.sidebar:
        st.markdown(
            f"<div class='fl-user'>Signed in as <b>{html.escape(u['username'])}</b>"
            f"<span class='fl-role'>{html.escape(u['role'])}</span></div>",
            unsafe_allow_html=True,
        )
        if st.button("Sign out", key="sign_out"):
            get_db().log_admin(u["username"], "logout", u["username"])
            logout("You have signed out.")
            st.rerun()


def page_setup(title: str, subtitle: str | None = None, permission: str = "view"):
    with contextlib.suppress(Exception):  # already set by the navigation entry point
        st.set_page_config(page_title=f"FraudLens - {title}", layout="wide")
    require_login()
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='fl-muted'>{html.escape(subtitle)}</div>", unsafe_allow_html=True)
    require_permission(permission)


# ----------------------------------------------------------------------------- artifact loaders
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def artifacts_ready() -> bool:
    return all(
        (ARTIFACTS / f).exists() for f in ["model.joblib", "metrics.json", "test_scored.parquet", "history.parquet"]
    )


def require_artifacts():
    if not artifacts_ready():
        msg = "No trained model found in artifacts/. "
        msg += (
            "Use the Training Lab page or run `python -m fraudkit.train --config configs/default.yaml`."
            if allowed("retrain")
            else "Ask an administrator to train the model."
        )
        st.warning(msg)
        sidebar_user()
        st.stop()


@st.cache_resource(show_spinner="Loading model")
def _model(mtime: float, bands_json: str) -> FraudModel:
    return load_serving_model(ARTIFACTS / "model.joblib", bands=json.loads(bands_json))


def get_model() -> FraudModel:
    bands = json.dumps(get_settings().bands.model_dump(), sort_keys=True)
    return _model(_mtime(ARTIFACTS / "model.joblib"), bands)


def model_key() -> str:
    """Changes whenever the served model file changes (use it in cache keys)."""
    return str(_mtime(ARTIFACTS / "model.joblib"))


@st.cache_data(show_spinner=False)
def _metrics(mtime: float) -> dict:
    return load_metrics(ARTIFACTS / "metrics.json")


def get_metrics() -> dict:
    return _metrics(_mtime(ARTIFACTS / "metrics.json"))


@st.cache_data(show_spinner=False)
def _profile(mtime: float) -> dict | None:
    p = ARTIFACTS / "reference_profile.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def get_reference_profile() -> dict | None:
    return _profile(_mtime(ARTIFACTS / "reference_profile.json"))


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


# ----------------------------------------------------------------------------- sidebar
def _persisted_widget(name: str, default):
    """Streamlit drops a widget's state when page navigation re-creates it. Keep the value under a separate
    key ("pref_*") and re-seed the widget key ("w_*") from it on every run, so settings survive page switches."""
    ss = st.session_state
    ss.setdefault(f"pref_{name}", default)
    ss[f"w_{name}"] = ss[f"pref_{name}"]
    return f"w_{name}"


def _save_pref(name: str):
    st.session_state[f"pref_{name}"] = st.session_state[f"w_{name}"]


def sidebar_controls() -> tuple[float, float]:
    """Threshold and review cost shared by all pages (persisted across page switches)."""
    m = get_metrics()
    validated = float(m["threshold"]["value"])
    default_cost = float(m["threshold"]["review_cost"])
    ss = st.session_state
    with st.sidebar:
        st.markdown("### Decision settings")
        st.checkbox(
            "Use validated threshold",
            key=_persisted_widget("use_validated", True),
            on_change=_save_pref,
            args=("use_validated",),
            help="The threshold that minimised missed-fraud + review cost on the validation period.",
        )
        if ss["pref_use_validated"]:
            thr = validated
            st.markdown(
                f"<div class='fl-muted'>Threshold: <b>{validated:.4f}</b> (chosen on validation)</div>",
                unsafe_allow_html=True,
            )
        else:
            thr = float(
                st.slider(
                    "Flag threshold (calibrated probability)",
                    0.001,
                    0.99,
                    step=0.001,
                    key=_persisted_widget("custom_threshold", max(0.001, round(validated, 3))),
                    on_change=_save_pref,
                    args=("custom_threshold",),
                    format="%.3f",
                )
            )
        st.number_input(
            "Review cost per flagged transaction (€)",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            key=_persisted_widget("review_cost", default_cost),
            on_change=_save_pref,
            args=("review_cost",),
        )
        version = m["run"].get("model_version", "unversioned")
        st.caption(f"Model {m['selected_model']} · version {version}")
    sidebar_user()
    return thr, float(ss["pref_review_cost"])


# ----------------------------------------------------------------------------- helpers
def business_metrics(scored: pd.DataFrame, thr: float, review_cost: float) -> dict:
    return confusion_at(
        scored["is_fraud"].to_numpy(), scored["p"].to_numpy(), scored["amt"].to_numpy(), thr, review_cost
    )


def money(x: float, decimals: int = 0) -> str:
    return f"€{x:,.{decimals}f}"


def money_short(x: float) -> str:
    """Compact money for KPI tiles (exact value goes in the tooltip)."""
    a = abs(x)
    if a >= 1e6:
        return f"€{x / 1e6:.2f}M"
    if a >= 1e4:
        return f"€{x / 1e3:.1f}k"
    return f"€{x:,.0f}"


def band_chip(band: str) -> str:
    color = BAND_COLORS.get(band, SLATE)
    return f"<span class='fl-chip' style='background:{color}'>{html.escape(str(band))}</span>"


def rule_cards(rules) -> str:
    out = []
    for r in rules:
        out.append(
            f"<div class='fl-card rule'><div class='t'><b>Policy rule</b> &middot; {html.escape(r.text)}</div>"
            f"<div class='w'>raises the band to at least {html.escape(r.band)}</div></div>"
        )
    return "".join(out)


def reason_cards(reasons, kind: str) -> str:
    if not reasons:
        return "<div class='fl-muted'>None above the reporting cut-off.</div>"
    out = []
    for r in reasons:
        out.append(
            f"<div class='fl-card {kind}'><div class='t'>{html.escape(r.text)}</div>"
            f"<div class='w'>{'+' if r.contribution > 0 else ''}{r.contribution:.2f} log-odds</div></div>"
        )
    return "".join(out)


def gauge(p: float, threshold: float, bounds: dict) -> go.Figure:
    steps = [
        dict(range=[0, bounds["medium"] * 100], color="#DCFCE7"),
        dict(range=[bounds["medium"] * 100, bounds["high"] * 100], color="#FEF9C3"),
        dict(range=[bounds["high"] * 100, bounds["critical"] * 100], color="#FFEDD5"),
        dict(range=[bounds["critical"] * 100, 100], color="#FEE2E2"),
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number=dict(suffix="%", valueformat=".1f"),
            gauge=dict(
                axis=dict(range=[0, 100], ticksuffix="%"),
                bar=dict(color="#0F172A", thickness=0.25),
                steps=steps,
                threshold=dict(line=dict(color=BLUE, width=3), thickness=0.85, value=threshold * 100),
            ),
            title=dict(text="Calibrated fraud probability", font=dict(size=14)),
        )
    )
    fig.update_layout(height=250, margin=dict(l=40, r=40, t=50, b=10))
    return fig


# (path, title, permission needed to see the page in the navigation)
PAGES = [
    ("views/1_Dashboard.py", "Dashboard", "view"),
    ("views/2_Check_Transaction.py", "Check a transaction", "view"),
    ("views/3_Batch_Scoring.py", "Batch scoring", "score"),
    ("views/4_Case_Queue.py", "Case queue", "view"),
    ("views/7_Monitoring.py", "Monitoring", "view"),
    ("views/5_Model_Report.py", "Model report", "view"),
    ("views/6_Training_Lab.py", "Training Lab", "view"),
    ("views/8_Admin.py", "Admin", "manage_users"),
    ("views/9_Account.py", "My account", "view"),
]


def pages_list():
    return [(p, t) for p, t, perm in PAGES if allowed(perm)]
