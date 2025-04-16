import streamlit as st
import subprocess
import sys

st.set_page_config(page_title="Bond Robo-Adviser Portal", layout="centered")

st.title("🤖 Welcome to Bond Robo-Adviser Platform")

st.markdown("""
Choose a module to proceed:
""")

col1, col2 = st.columns(2)

with col1:
    if st.button(" Part 1: Efficient Frontier"):
        subprocess.Popen(["streamlit", "run", "app.py"])
        st.success("Launching Part 1...")

with col2:
    if st.button(" Part 2: Risk Aversion Optimizer"):
        subprocess.Popen(["streamlit", "run", "risk_aversion_app.py"])
        st.success("Launching Part 2...")