"""FraudLens entry point: `streamlit run app/streamlit_app.py` (run from the fraudlens/ folder)."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import pages_list  # noqa: E402

st.set_page_config(page_title="FraudLens", layout="wide")
nav = st.navigation([st.Page(path, title=title) for path, title in pages_list()])
st.sidebar.markdown("**FraudLens** &middot; explainable card-fraud scoring")
nav.run()
