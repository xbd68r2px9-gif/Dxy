"""
app.py
------
Streamlit UI — display only, no business logic here.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd

from logic import get_last_updated, generate_sample_data, compute_summary

# ── Page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(page_title="Minimal App", layout="centered")

st.title("Minimal Streamlit App")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    days = st.slider("Days of data", min_value=7, max_value=90, value=30)

# ── Load data with error handling ────────────────────────────────────────────
try:
    data = generate_sample_data(days=days)
    summary = compute_summary(data)
except ValueError as e:
    st.error(f"Data error: {e}")
    st.stop()                          # halt the app here if something went wrong
except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.stop()

# ── Last-updated timestamp ───────────────────────────────────────────────────
st.caption(f"Last updated: {get_last_updated()}")

# ── Output block: summary stats ──────────────────────────────────────────────
st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Count", summary["count"])
col2.metric("Min", summary["min"])
col3.metric("Max", summary["max"])
col4.metric("Mean", summary["mean"])

# ── Chart ────────────────────────────────────────────────────────────────────
st.subheader("Daily Values")
df = pd.DataFrame(data)               # convert list-of-dicts → DataFrame
df["date"] = pd.to_datetime(df["date"])
st.line_chart(df.set_index("date")["value"])
