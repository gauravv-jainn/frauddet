import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from common import (
    ROOT,
    allowed,
    artifacts_ready,
    get_api_keys,
    get_case_store,
    get_db,
    get_metrics,
    get_settings,
    get_users,
    page_setup,
    sidebar_controls,
    sidebar_user,
    username,
)
from fraudkit.auth import ROLES, AuthError
from fraudkit.score import ARTIFACTS
from fraudkit.security import sanitize_csv_frame

page_setup("Admin", "Users, API keys, audit log and system status.", permission="manage_users")
if artifacts_ready():
    sidebar_controls()
else:
    sidebar_user()
me = username()
users, keys, db = get_users(), get_api_keys(), get_db()

t_users, t_keys, t_audit, t_system = st.tabs(["Users", "API keys", "Audit log", "System"])

# ----------------------------------------------------------------------------- users
with t_users:
    ul = users.list()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "username": u.username,
                    "role": u.role,
                    "active": u.active,
                    "must change password": u.must_change_pw,
                    "last login": u.last_login,
                    "created": u.created_at,
                }
                for u in ul
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    left, right = st.columns(2)
    with left, st.form("new_user", clear_on_submit=True):
        st.markdown("**Add a user**")
        nu = st.text_input("Username")
        nr = st.selectbox(
            "Role",
            ROLES,
            index=1,
            help="viewer: read-only · analyst: score and work cases · admin: everything incl. users, keys, retraining",
        )
        npw = st.text_input(
            "Temporary password", type="password", help="The user must choose their own password at first sign-in."
        )
        if st.form_submit_button("Create user", type="primary"):
            try:
                users.create(nu, npw, nr, actor=me, must_change_pw=True)
                st.success(
                    f"User '{nu.strip().lower()}' created. Share the temporary password through a separate channel."
                )
                st.rerun()
            except AuthError as e:
                st.error(str(e))
    with right:
        st.markdown("**Manage a user**")
        target = st.selectbox("User", [u.username for u in ul], key="admin_target")
        tu = users.get(target)
        new_role = st.selectbox("Role", ROLES, index=ROLES.index(tu.role), key=f"role_{target}")
        b1, b2 = st.columns(2)
        if b1.button("Save role", key=f"save_role_{target}", disabled=new_role == tu.role):
            try:
                users.set_role(target, new_role, actor=me)
                st.rerun()
            except AuthError as e:
                st.error(str(e))
        label = "Disable account" if tu.active else "Enable account"
        if b2.button(label, key=f"toggle_{target}"):
            try:
                users.set_active(target, not tu.active, actor=me)
                st.rerun()
            except AuthError as e:
                st.error(str(e))
        with st.form(f"reset_{target}", clear_on_submit=True):
            rpw = st.text_input("New temporary password", type="password")
            if st.form_submit_button("Reset password"):
                try:
                    users.set_password(target, rpw, actor=me, must_change=True)
                    st.success("Password reset; the user must change it at next sign-in.")
                except AuthError as e:
                    st.error(str(e))

# ----------------------------------------------------------------------------- api keys
with t_keys:
    if not allowed("manage_api_keys"):
        st.info("Your role cannot manage API keys.")
    else:
        if token := st.session_state.pop("new_api_key", None):
            st.warning("Copy this key now. It is stored only as a hash and cannot be shown again.")
            st.code(token, language=None)
        with st.form("new_key", clear_on_submit=True):
            kn = st.text_input("Key name (who or what will use it)", placeholder="e.g. checkout-service-staging")
            if st.form_submit_button("Create API key", type="primary"):
                try:
                    _, tok = keys.create(kn, actor=me)
                    st.session_state["new_api_key"] = tok
                    st.rerun()
                except AuthError as e:
                    st.error(str(e))
        kl = keys.list()
        if kl:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "key id": k.key_id,
                            "name": k.name,
                            "active": k.active,
                            "created by": k.created_by,
                            "created": k.created_at,
                            "last used": k.last_used,
                            "requests": k.use_count,
                            "revoked": k.revoked_at,
                        }
                        for k in kl
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            active = [k.key_id for k in kl if k.active]
            if active:
                r1, r2 = st.columns([2, 1])
                rk = r1.selectbox("Revoke key", active, format_func=lambda i: f"{i} ({keys.get(i).name})")
                if r2.button("Revoke", type="primary"):
                    keys.revoke(rk, actor=me)
                    st.rerun()
        st.caption("Clients send the key in the X-API-Key header to POST /v1/score. See the API section of the README.")

# ----------------------------------------------------------------------------- audit
with t_audit:
    with db.tx() as c:
        admin_log = pd.read_sql_query(
            "SELECT at, actor, action, target, detail FROM admin_audit ORDER BY id DESC LIMIT 2000", c
        )
    f1, f2 = st.columns(2)
    actors = f1.multiselect("Actor", sorted(admin_log["actor"].dropna().unique()))
    actions = f2.multiselect("Action", sorted(admin_log["action"].unique()))
    view = admin_log
    if actors:
        view = view[view["actor"].isin(actors)]
    if actions:
        view = view[view["action"].isin(actions)]
    st.markdown("**Security and admin events**")
    st.dataframe(view, hide_index=True, width="stretch", height=320)
    st.download_button(
        "Export admin log (CSV)",
        sanitize_csv_frame(view).to_csv(index=False).encode(),
        "fraudlens_admin_audit.csv",
        "text/csv",
    )
    st.markdown("**Case decisions**")
    case_audit = get_case_store().audit()
    st.dataframe(case_audit, hide_index=True, width="stretch", height=280)
    st.download_button(
        "Export case audit (CSV)",
        sanitize_csv_frame(case_audit).to_csv(index=False).encode(),
        "fraudlens_case_audit.csv",
        "text/csv",
    )

# ----------------------------------------------------------------------------- system
with t_system:
    cfg = get_settings()
    c = st.columns(3)
    if artifacts_ready():
        m = get_metrics()
        c[0].metric("Served model", m["run"].get("model_version", "unversioned"))
        c[1].metric("Decision threshold", f"{m['threshold']['value']:.4f}")
    c[2].metric("Database schema", f"v{db.schema_version}")
    files = []
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            files.append(
                {
                    "file": f.name,
                    "size MB": round(f.stat().st_size / 1e6, 2),
                    "modified": pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"),
                }
            )
    st.markdown("**Artifacts**")
    st.dataframe(pd.DataFrame(files), hide_index=True, width="stretch")
    runs = sorted((ARTIFACTS / "runs").glob("*/metrics.json"), reverse=True)
    if runs:
        st.markdown("**Model versions (training runs)**")
        hist = []
        for p in runs:
            mm = json.loads(p.read_text(encoding="utf-8"))
            hist.append(
                {
                    "version": p.parent.name,
                    "fast run": mm["run"].get("fast"),
                    "test PR-AUC": round(mm["test"]["pr_auc"], 4),
                    "recall": round(mm["test"]["at_threshold"]["recall"], 4),
                    "threshold": round(mm["threshold"]["value"], 4),
                }
            )
        st.dataframe(pd.DataFrame(hist), hide_index=True, width="stretch")
    st.markdown("**Configuration** (configs/default.yaml, validated at start-up)")
    st.json(cfg.model_dump(), expanded=False)
    st.caption(f"Database: {cfg.db_path().relative_to(ROOT) if cfg.db_path().is_relative_to(ROOT) else cfg.db_path()}")
