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
ACTION_COLORS = {"approve": "#15803D", "step_up": "#CA8A04", "review": "#EA580C", "block": "#B91C1C"}
ACTION_LABELS = {"approve": "Approve", "step_up": "Step-up", "review": "Review", "block": "Block"}

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
:root { --fl-ink:#0F172A; --fl-muted:#64748B; --fl-line:#E2E8F0; --fl-soft:#F8FAFC; --fl-brand:#1D4ED8; }
.block-container {padding-top: 3.4rem; max-width: 1280px;}
h1 {font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -0.01em;}
h2, h3 {font-weight: 650 !important; letter-spacing: -0.005em;}
div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid var(--fl-line); border-radius: 12px;
    padding: 12px 14px; box-shadow: 0 1px 2px rgba(15,23,42,.04);}
div[data-testid="stMetricLabel"] p {color: var(--fl-muted); font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: .04em; font-weight: 600;}
div[data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 700; color: var(--fl-ink);}
.fl-header {display:flex; justify-content:space-between; align-items:flex-end; gap:16px;
    border-bottom:1px solid var(--fl-line); padding-bottom:10px; margin-bottom:14px;}
.fl-header h1 {margin:0 !important; padding:0 !important; line-height:1.25 !important;}
.fl-header .sub {color: var(--fl-muted); font-size: .9rem; margin-top: 2px;}
.fl-header .badges {display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end;}
.fl-pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:600;
    background:#EEF2FF; color:#1E3A8A; border:1px solid #C7D2FE; white-space:nowrap;}
.fl-pill.grey {background:#F1F5F9; color:#334155; border-color:#E2E8F0;}
.fl-card {border: 1px solid var(--fl-line); border-left-width: 5px; border-radius: 10px; padding: 10px 14px;
    margin-bottom: 8px; background: #FFFFFF;}
.fl-card.raise {border-left-color: #B91C1C; background: #FEF2F2;}
.fl-card.lower {border-left-color: #15803D; background: #F0FDF4;}
.fl-card.rule {border-left-color: #7C3AED; background: #F5F3FF;}
.fl-card.note {border-left-color: #64748B; background: #F8FAFC;}
.fl-card .t {font-size: 0.93rem; color: var(--fl-ink);}
.fl-card .w {font-size: 0.74rem; color: var(--fl-muted); margin-top: 2px;}
.fl-chip {display: inline-block; padding: 3px 12px; border-radius: 999px; color: white; font-weight: 650;
    font-size: 0.82rem;}
.fl-muted {color: var(--fl-muted); font-size: 0.88rem;}
.fl-action {border: 1px solid #CBD5E1; border-radius: 10px; padding: 10px 14px; background: var(--fl-soft);}
.fl-lane {border:1px solid var(--fl-line); border-radius:12px; padding:10px; background:var(--fl-soft);
    min-height:120px;}
.fl-lane h4 {margin:0 0 8px 0; font-size:.85rem; color:#334155; text-transform:uppercase; letter-spacing:.04em;}
.fl-case {background:#FFFFFF; border:1px solid var(--fl-line); border-radius:10px; padding:8px 10px; margin-bottom:6px;
    font-size:.82rem;}
.fl-case b {color: var(--fl-ink);}
.fl-case .meta {color: var(--fl-muted); font-size:.74rem;}
.fl-overdue {color:#B91C1C; font-weight:700;}
section[data-testid="stSidebar"] .fl-muted {color:#94A3B8;}
.fl-user {font-size: 0.85rem;}
.fl-role {display: inline-block; padding: 1px 8px; border-radius: 6px; background: #E0E7FF; color: #1E3A8A;
    font-size: 0.72rem; font-weight: 700; margin-left: 4px;}
.fl-inbox {border:1px solid rgba(148,163,184,.35); border-radius:10px; padding:8px 10px; margin:6px 0;
    font-size:.82rem;}
</style>
"""


# ----------------------------------------------------------------------------- config / stores
@st.cache_resource
def get_settings() -> Settings:
    return load_config()


def get_config() -> dict:
    return get_settings().model_dump()


def get_watchlist():
    from fraudkit.watchlist import WatchlistStore

    return WatchlistStore(get_db())


def policy(decision: dict | None = None) -> dict:
    """Everything an explanation needs to decide: resolved rule limits, training ranges, the decision policy
    (sidebar / validated) and the live watchlists (blocked cards, watched merchants). See fraudkit/rules.py."""
    from fraudkit.monitoring import training_ranges
    from fraudkit.rules import resolve

    return {
        "rules": resolve(get_settings().rules.model_dump(), get_metrics()),
        "ranges": training_ranges(get_reference_profile()),
        "decision": decision or current_policy(),
        "watch": get_watchlist().active_sets(),
    }


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
        st.divider()
        st.markdown(
            f"<div class='fl-user'>Signed in as <b>{html.escape(u['username'])}</b>"
            f"<span class='fl-role'>{html.escape(u['role'])}</span></div>",
            unsafe_allow_html=True,
        )
        sla = get_settings().decision.sla_hours
        s = get_case_store().stats(sla, u["username"])
        if s["total"]:
            over = f" &middot; <span class='fl-overdue'>{s['my_overdue']} overdue</span>" if s["my_overdue"] else ""
            st.markdown(
                f"<div class='fl-inbox'>Inbox: <b>{s['my_open']}</b> assigned to you{over}<br>"
                f"<span class='fl-muted'>{s['open']} open in the queue &middot; {s['overdue']} past the "
                f"{sla:g} h SLA</span></div>",
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
    badges = ""
    if artifacts_ready():
        m = get_metrics()
        mode = (m.get("decision") or {}).get("served", "threshold").replace("_", " ")
        badges = (
            f"<span class='fl-pill'>model {html.escape(m['run'].get('model_version', 'unversioned'))}</span>"
            f"<span class='fl-pill grey'>validated policy: {html.escape(mode)}</span>"
        )
    sub = f"<div class='sub'>{html.escape(subtitle)}</div>" if subtitle else ""
    st.markdown(
        f"<div class='fl-header'><div><h1>{html.escape(title)}</h1>{sub}</div><div class='badges'>{badges}</div></div>",
        unsafe_allow_html=True,
    )
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


def current_policy() -> dict:
    """Decision policy from the sidebar (validated by default)."""
    from fraudkit.decision import serving_policy

    m = get_metrics()
    ss = st.session_state
    critical = get_settings().bands.critical
    if ss.get("pref_use_validated", True):
        return serving_policy(m, get_settings().decision.mode, critical=critical)
    mode = "expected_loss" if ss.get("pref_policy_mode", "Expected loss") == "Expected loss" else "threshold"
    return serving_policy(
        m,
        mode,
        review_cost=ss.get("pref_review_cost"),
        threshold=ss.get("pref_custom_threshold"),
        k=ss.get("pref_custom_k"),
        critical=critical,
    )


def sidebar_controls() -> dict:
    """Decision policy shared by all pages (persisted across page switches). Returns the policy dict:
    mode, k, threshold, review_cost, critical (see fraudkit/decision.py)."""
    from fraudkit.decision import describe_policy

    m = get_metrics()
    validated = float(m["threshold"]["value"])
    default_cost = float(m["threshold"]["review_cost"])
    default_k = float((m.get("decision") or {}).get("expected_loss", {}).get("k", 1.0))
    ss = st.session_state
    with st.sidebar:
        st.markdown("##### Decision policy")
        st.checkbox(
            "Use validated policy",
            key=_persisted_widget("use_validated", True),
            on_change=_save_pref,
            args=("use_validated",),
            help="The policy and parameters that minimised missed fraud + review cost on the validation period.",
        )
        if not ss["pref_use_validated"]:
            st.radio(
                "Policy",
                ["Expected loss", "Flat threshold"],
                key=_persisted_widget("policy_mode", "Expected loss"),
                on_change=_save_pref,
                args=("policy_mode",),
                horizontal=True,
            )
            if ss["pref_policy_mode"] == "Expected loss":
                st.slider(
                    "k: review when p x amount >= k x review cost",
                    0.25,
                    10.0,
                    step=0.25,
                    key=_persisted_widget("custom_k", default_k),
                    on_change=_save_pref,
                    args=("custom_k",),
                )
            else:
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
            st.number_input(
                "Review cost per flagged transaction (€)",
                min_value=0.0,
                max_value=500.0,
                step=1.0,
                key=_persisted_widget("review_cost", default_cost),
                on_change=_save_pref,
                args=("review_cost",),
            )
        pol = current_policy()
        st.markdown(f"<div class='fl-muted'>{html.escape(describe_policy(pol))}</div>", unsafe_allow_html=True)
    sidebar_user()
    return pol


# ----------------------------------------------------------------------------- helpers
def business_metrics(scored: pd.DataFrame, pol: dict) -> dict:
    """Money metrics of the model-side decision policy (equals metrics.json -> test.at_policy when validated)."""
    from fraudkit.decision import policy_flags

    flags = policy_flags(scored["p"].to_numpy(), scored["amt"].to_numpy(), pol)
    m = confusion_at(
        scored["is_fraud"].to_numpy(), flags.astype(float), scored["amt"].to_numpy(), 0.5, pol["review_cost"]
    )
    risky = scored["p"].to_numpy() >= get_model().band_bounds(pol["threshold"])["medium"]
    m["step_up"] = int((risky & ~flags).sum())
    return m


def decisions_vector(scored: pd.DataFrame, pol: dict) -> np.ndarray:
    """approve / step_up / review / block for every row, model-side policy only (fast, vectorised)."""
    from fraudkit.decision import policy_flags

    p = scored["p"].to_numpy()
    flags = policy_flags(p, scored["amt"].to_numpy(), pol)
    medium = get_model().band_bounds(pol["threshold"])["medium"]
    return np.where(
        p >= pol["critical"], "block", np.where(flags, "review", np.where(p >= medium, "step_up", "approve"))
    )


def action_badge(decision: str) -> str:
    color = ACTION_COLORS.get(decision, SLATE)
    label = ACTION_LABELS.get(decision, decision)
    return f"<span class='fl-chip' style='background:{color}'>{html.escape(label)}</span>"


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


def gauge(score: int, threshold_score: int, bound_scores: dict) -> go.Figure:
    """Risk score gauge (0-999). Band colours mark where Medium / High / Critical probabilities fall on the scale."""
    steps = [
        dict(range=[0, bound_scores["medium"]], color="#DCFCE7"),
        dict(range=[bound_scores["medium"], bound_scores["high"]], color="#FEF9C3"),
        dict(range=[bound_scores["high"], bound_scores["critical"]], color="#FFEDD5"),
        dict(range=[bound_scores["critical"], 999], color="#FEE2E2"),
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(valueformat="d"),
            gauge=dict(
                axis=dict(range=[0, 999], tickvals=[0, 250, 500, 750, 999]),
                bar=dict(color="#0F172A", thickness=0.25),
                steps=steps,
                threshold=dict(line=dict(color=BLUE, width=3), thickness=0.85, value=threshold_score),
            ),
            title=dict(text="Risk score (0-999)", font=dict(size=14)),
        )
    )
    fig.update_layout(height=250, margin=dict(l=40, r=40, t=50, b=10))
    return fig


def score_bounds(fm, threshold: float) -> tuple[int, dict]:
    b = fm.band_bounds(threshold)
    sc = {k: int(fm.risk_score(v)[0]) for k, v in b.items()}
    return int(fm.risk_score(threshold)[0]), sc


# (section, path, title, icon, permission needed to see the page in the navigation)
PAGES = [
    ("Overview", "views/1_Dashboard.py", "Dashboard", ":material/space_dashboard:", "view"),
    ("Overview", "views/7_Monitoring.py", "Monitoring", ":material/monitor_heart:", "view"),
    ("Investigate", "views/4_Case_Queue.py", "Case queue", ":material/inbox:", "view"),
    ("Investigate", "views/2_Check_Transaction.py", "Check a transaction", ":material/search:", "view"),
    ("Investigate", "views/10_Card_360.py", "Card 360", ":material/credit_card:", "view"),
    ("Investigate", "views/3_Batch_Scoring.py", "Batch scoring", ":material/upload_file:", "score"),
    ("Model", "views/5_Model_Report.py", "Model report", ":material/analytics:", "view"),
    ("Model", "views/6_Training_Lab.py", "Training Lab", ":material/model_training:", "view"),
    ("Settings", "views/8_Admin.py", "Admin", ":material/admin_panel_settings:", "manage_users"),
    ("Settings", "views/9_Account.py", "My account", ":material/person:", "view"),
]


def pages_list():
    """[(path, title)] the current user may see."""
    return [(p, t) for _, p, t, _, perm in PAGES if allowed(perm)]


def navigation_sections() -> dict:
    out: dict[str, list] = {}
    for section, path, title, icon, perm in PAGES:
        if allowed(perm):
            out.setdefault(section, []).append(st.Page(path, title=title, icon=icon))
    return out


def open_card(cc_num):
    """Jump to the Card 360 page for a card."""
    st.session_state["card360_card"] = int(cc_num)
    st.switch_page("views/10_Card_360.py")


@st.cache_data(show_spinner=False)
def _card_refs(mtime: float) -> dict:
    from fraudkit.security import card_hash

    return {card_hash(c): int(c) for c in get_history()["cc_num"].unique()}


def card_from_ref(card_ref: str | None):
    """cc_num for a hashed card reference (cases store only the hash and the last four digits)."""
    if not card_ref:
        return None
    return _card_refs(_mtime(ARTIFACTS / "history.parquet")).get(card_ref)
