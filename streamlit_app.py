"""Streamlit entry point for the GRAG terminal-interface equivalents."""

from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="GRAG Router", page_icon="GR", layout="wide")

pages = [
    st.Page("ui_pages/home.py", title="GRAG Router", default=True),
    st.Page("ui_pages/fast.py", title="Fast", icon=":material/bolt:", url_path="fast"),
    st.Page("ui_pages/nrml.py", title="Normal", icon=":material/account_tree:", url_path="nrml"),
]

st.navigation(pages, position="hidden").run()
