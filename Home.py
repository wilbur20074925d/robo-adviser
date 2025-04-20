import streamlit as st

# ---- Page Configuration ----
st.set_page_config(page_title="ETF Robo-Adviser Portal", layout="centered")

# ---- Header ----
st.title("Welcome to the ETF Robo-Adviser Platform")

# ---- Introduction Text ----
st.markdown("""
Welcome to your **academic robo-adviser** — a smart tool to help you explore and understand optimal portfolio construction using ETFs.

This platform offers two interactive modules:

### Part 1: Efficient Frontier Visualizer  
Explore thousands of simulated portfolios, visualize the efficient frontier, and understand the trade-off between risk and return.

### Part 2: Risk Aversion Portfolio Optimizer  
Input your personal risk aversion level to generate a customized optimal portfolio and track how your preferences affect expected return and utility.

---

Use the **sidebar** on the left to navigate between modules and begin your journey into portfolio optimization!
""")
