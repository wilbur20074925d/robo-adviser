import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 

st.set_page_config(page_title="Optimal Portfolio Based on Risk Aversion", layout="wide")
st.title(" Optimal Portfolio Calculator – Risk Aversion-Based")

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
uploaded_file = st.file_uploader("Upload bond ETF daily price CSV (same format as Part 1)", type=["csv"])

# Questionnaire inputs
allow_short = st.toggle("Allow Short Sales in Optimal Portfolio?", value=True)

st.header("📋 Investor Questionnaire")
Q1 = st.slider("1. How would you react if your portfolio dropped 10% in a month?", 1, 10, 5,
               help="1 = Panic and sell all, 10 = Stay calm or buy more")
Q2 = st.slider("2. What type of return are you aiming for?", 1, 10, 5,
               help="1 = Stable low returns, 10 = Aggressive growth")
Q3 = st.slider("3. Do you plan to use this money within the next 5 years?", 1, 10, 5,
               help="1 = Yes definitely, 10 = No plans at all")
Q4 = st.slider("4. What is your investment experience level?", 1, 10, 5,
               help="1 = No experience, 10 = More than 10 years")
Q5 = st.slider("5. How do you feel about day-to-day fluctuations in your portfolio?", 1, 10, 5,
               help="1 = Anxious or stressed, 10 = Indifferent")
Q6 = st.slider("6. What is your primary investment objective?", 1, 10, 5,
               help="1 = Preserve capital, 10 = Maximize long-term capital gains")

# Reverse-mapped scoring logic
A_raw = (
    0.25 * (10 - Q1) +
    0.20 * (10 - Q2) +
    0.15 * (10 - Q3) +
    0.15 * (10 - Q4) +
    0.15 * (10 - Q5) +
    0.10 * (10 - Q6)
)

A = round(max(A_raw, 0.5), 2)
# Optional: categorize risk profile
if A >= 6:
    profile = "Conservative"
elif A >= 3:
    profile = "Balanced"
else:
    profile = "Aggressive"

st.markdown(f"### Estimated Risk Aversion Coefficient: **A = {A}**")
st.markdown(f"### Risk Profile: **{profile}**")

# Button to trigger calculation
if uploaded_file and st.button("Calculate Optimal Portfolio"):
    # Load and clean data
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.dropna()

    st.success("File uploaded and processed successfully.")

    st.subheader("Sample of Uploaded Data")
    st.dataframe(df.head())

    # Portfolio statistics
    returns = df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    mean_vec = mean_returns.values  # keep for later reuse
    cov_inv = np.linalg.inv(cov_matrix.values)
    mu = mean_vec
    sigma = cov_matrix.values
    n_assets = len(mu)

    # Objective: maximize U = w^T mu - 0.5 * A * w^T Σ w
    def neg_utility(w, mu, sigma, A):
        port_return = np.dot(w, mu)
        port_var = np.dot(w.T, np.dot(sigma, w))
        return -(port_return - 0.5 * A * port_var)  # minimize negative utility

    # Constraints: sum(weights) == 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: allow/disallow short sales
    bounds = [(-1, 1)] * n_assets if allow_short else [(0, 1)] * n_assets

    # Initial guess: equal weights
    init_guess = np.ones(n_assets) / n_assets

    # Optimization
    result = opt.minimize(neg_utility, init_guess, args=(mu, sigma, A), method='SLSQP',
                        bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x
    else:
        st.error("Optimization failed.")
        st.stop()

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
    st.subheader("Optimal Portfolio Based on Your Risk Aversion")

    weights_df = pd.DataFrame({
        'Fund': mean_returns.index,
        'Weight': optimal_weights.flatten()
    }).set_index("Fund")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.dataframe(weights_df.style.format("{:.2%}"))

    with col2:
        if (weights_df["Weight"] < 0).any():
            st.warning("Your optimal portfolio includes short positions. Negative weights are excluded from the pie chart.")

        weights_nonneg = weights_df.copy()
        weights_nonneg["Weight"] = weights_nonneg["Weight"].clip(lower=0)

        # Filter out negligible weights for better display
        plot_df = weights_nonneg[weights_nonneg["Weight"] > 0.01]  # skip <1% weights

        # Handle edge case: all weights too small
        if plot_df.empty:
            st.warning("No positive weights to display in pie chart.")
        else:
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                plot_df["Weight"],
                labels=plot_df.index,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 8}
            )
            for text in texts + autotexts:
                text.set_fontweight('bold')
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
    st.subheader("Session History of Calculations")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.style.format({
        "Expected Return": "{:.2%}",
        "Volatility": "{:.2%}",
        "Utility": "{:.4f}"
    }))
