import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from common import (
    artifacts_ready,
    change_password_form,
    current_user,
    get_settings,
    get_users,
    page_setup,
    sidebar_controls,
    sidebar_user,
)
from fraudkit.auth import MIN_PASSWORD_LEN, PERMISSIONS

page_setup("My account")
if artifacts_ready():
    sidebar_controls()
else:
    sidebar_user()
sess = current_user()
user = get_users().get(sess["username"])

if st.session_state.pop("pw_changed", False):
    st.success("Password changed.")
c = st.columns(3)
c[0].metric("Username", user.username)
c[1].metric("Role", user.role)
c[2].metric("Session timeout", f"{get_settings().app.session_timeout_minutes} min")
st.caption(f"Last sign-in: {user.last_login} UTC · account created {user.created_at} UTC")
st.markdown(
    "**What your role can do:** "
    + ", ".join(sorted(p.replace("_", " ") for p, roles in PERMISSIONS.items() if user.role in roles))
)

st.subheader("Change password")
st.caption(f"At least {MIN_PASSWORD_LEN} characters with a letter and a digit, not containing your username.")
change_password_form("own_pw")
