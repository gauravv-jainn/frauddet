"""FraudLens entry point: `streamlit run app/streamlit_app.py` (run from the fraudlens/ folder)."""

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import navigation_sections, require_login

st.set_page_config(page_title="FraudLens", page_icon=str(HERE / "assets" / "icon.svg"), layout="wide")
st.logo(str(HERE / "assets" / "logo.svg"), icon_image=str(HERE / "assets" / "icon.svg"), size="large")
require_login()  # login / first-run setup; stops here when nobody is signed in
nav = st.navigation(navigation_sections())
nav.run()
