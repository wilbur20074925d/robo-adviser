import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimal Portfolio Based on Risk Aversion", layout="wide")
st.title("🧠 Optimal Portfolio Calculator – Risk Aversion-Based")

st.markdown("""
This tool computes your optimal portfolio based on your risk aversion level using the utility function:

\[
U = r - \frac{A \cdot \sigma^2}{2}
\]

Where:
- \( r \) = expected portfolio return  
- \( \sigma^2 \) = portfolio variance  
- \( A \) = risk aversion coefficient
""")

# Initialize session state to store history
if 'history' not in st.session_state:
    st.session_state.history = []

# Upload price data
uploaded_file = st.file_uploader("📤 Upload bond ETF daily price CSV (same format as Part 1)", type=["csv"])

# Questionnaire inputs
allow_short = st.toggle("🟢 Allow Short Sales in Optimal Portfolio?", value=True)

st.header("📋 Investor Questionnaire")
q1 = st.slider("1. How would you feel if your portfolio lost 10% in a month?", 1, 10, 5)
q2 = st.slider("2. How important is stable income over high returns?", 1, 10, 5)
q3 = st.slider("3. How much market experience do you have?", 1, 10, 5)
q4 = st.slider("4. Are you investing for long-term growth?", 1, 10, 5)

A = (q1 + q2 - q3 + (10 - q4)) / 4
A = round(max(A, 0.1), 2)
st.markdown(f"### 🧮 Estimated Risk Aversion Coefficient: **A = {A}**")

# Button to trigger calculation
if uploaded_file and st.button("🔄 Calculate Optimal Portfolio"):
    # Load and clean data
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.dropna()

    st.success("✅ File uploaded and processed successfully.")

    st.subheader("📊 Sample of Uploaded Data")
    st.dataframe(df.head())

    # Portfolio statistics
    returns = df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Calculate optimal weights
    cov_inv = np.linalg.inv(cov_matrix.values)
    mean_vec = mean_returns.values.reshape(-1, 1)
    raw_weights = cov_inv @ mean_vec

    if allow_short:
        optimal_weights = raw_weights / np.sum(np.abs(raw_weights))
    else:
        raw_weights = np.maximum(raw_weights, 0)
        optimal_weights = raw_weights / np.sum(raw_weights)

    # Portfolio metrics
    port_return = float(optimal_weights.T @ mean_vec)
    port_var = float(optimal_weights.T @ cov_matrix.values @ optimal_weights)
    utility = port_return - 0.5 * A * port_var

    # Store result in session history
    st.session_state.history.append({
        'Risk Aversion A': A,
        'Allow Short Sales': allow_short,
        'Expected Return': port_return,
        'Volatility': np.sqrt(port_var),
        'Utility': utility
    })

    # Display results
    st.markdown("---")
    st.subheader("📊 Optimal Portfolio Based on Your Risk Aversion")

    weights_df = pd.DataFrame({
        'Fund': mean_returns.index,
        'Weight': optimal_weights.flatten()
    }).set_index("Fund")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.dataframe(weights_df.style.format("{:.2%}"))

    with col2:
        if (weights_df["Weight"] < 0).any():
            st.warning("⚠️ Your optimal portfolio includes short positions. Negative weights are excluded from the pie chart.")

        weights_nonneg = weights_df.copy()
        weights_nonneg["Weight"] = weights_nonneg["Weight"].clip(lower=0)

        fig, ax = plt.subplots()
        ax.pie(weights_nonneg['Weight'], labels=weights_nonneg.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    st.markdown(f"""
    - **Expected Return**: {port_return:.2%}  
    - **Portfolio Volatility**: {np.sqrt(port_var):.2%}  
    - **Utility Score (U)**: {utility:.4f}
    """)

# Always show history at the bottom
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Session History of Calculations")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.style.format({
        "Expected Return": "{:.2%}",
        "Volatility": "{:.2%}",
        "Utility": "{:.4f}"
    }))