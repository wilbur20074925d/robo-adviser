import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Optimal Portfolio Based on Risk Aversion", layout="wide")
st.title(" Optimal Portfolio Calculator - Risk Aversion-Based")
def simulate_portfolios(mean_returns, cov_matrix, n_portfolios=10000, allow_short=False):
    np.random.seed(42)
    n_assets = len(mean_returns)
    results = np.zeros((3, n_portfolios))

    for i in range(n_portfolios):
        if allow_short:
            w = np.random.randn(n_assets)
            # Make sure the sum of the weight is not 0, and all weight < 200%
            while abs(w.sum()) < 1e-6 or np.sum(np.abs(w / w.sum())) > 2 :
                w = np.random.randn(n_assets)
            weights = w / w.sum()
        else:
            w = np.random.rand(n_assets)
            weights = w / w.sum()
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(weights.T @ cov_matrix @ weights)
        results[0, i] = port_std
        results[1, i] = port_return
        results[2, i] = (port_return - 0.02) / port_std  # Sharpe Ratio with rf = 2%
    return results
st.markdown(r"""
### Optimal Portfolio Selection Based on Risk Aversion

This tool helps you determine your **optimal portfolio allocation** based on your individual **risk aversion level**, using the following utility maximization function:

$$
U = r - \frac{A \cdot \sigma^2}{2}
$$

Where:

- $U$ represents the **investor's utility** — a measure of satisfaction with a given risk-return trade-off  
- $r$ is the **expected return** of the portfolio  
- $\sigma^2$ is the **variance** of the portfolio (i.e., risk squared)  
- $A$ is the **risk aversion coefficient**, a positive number that reflects how much you dislike risk

---

This model assumes that you prefer **higher returns** and **lower risk**. The optimal portfolio is selected by maximizing your utility value $U$ given your personal risk tolerance.  
A **higher $A$** implies greater sensitivity to risk, leading to more conservative portfolios. Conversely, a **lower $A$** results in a more aggressive investment allocation.

Please adjust your risk aversion level using the slider below to see how your optimal weights change accordingly.
""")

# Initialize session state to store history
if 'history' not in st.session_state:
    st.session_state.history = []

if 'entries' not in st.session_state:
    st.session_state.entries = ['']
if 'calc_result' not in st.session_state:
    st.session_state.calc_result = None
if 'use_input' not in st.session_state:
    st.session_state.use_input = False

def add_entry():
    st.session_state.entries.append('')

def remove_entry(idx):
    st.session_state.entries.pop(idx)
    
# Loop through each etf and download 10 years of monthly adjusted close prices
def fetch_tickers(etf_tickers): 
    etf_data = {}
    for ticker in etf_tickers:
        print(f"Fetching data for {ticker}...")
        try:
            data = yf.Ticker(ticker).history(period="10y", interval="1d")
            if not data.empty:
                etf_price = data['Close']
                nav = etf_price / etf_price.iloc[0]
                etf_data[ticker] = nav
            else:
                print(f"No data returned for {ticker}. Skipping.")
        except Exception as e:
            print(f"Failed to get data for {ticker}. Reason: {e}")
            
    # Combine into a single DataFrame
    etf_prices_df = pd.DataFrame(etf_data)

    # Drop rows with any missing values (optional)
    etf_prices_df.dropna(inplace=True)

    return etf_prices_df

def calculate():
    etf_tickers = [st.session_state[f"entry_{i}"].strip() for i in range(len(st.session_state.entries))]
    
    if len(etf_tickers) == 0 or any(t == "" for t in etf_tickers):
        st.error("ETF Code cannot be empty!")
        return
    
    df = fetch_tickers(etf_tickers)
    st.success(f"Data Fetched")
    st.session_state.calc_result = df
    st.session_state.use_input = True

st.title("Input Your ETFs")

st.markdown("---")

for i, val in enumerate(st.session_state.entries):
    col1, col2 = st.columns([6, 1], gap="small")
    with col1:
        st.text_input(f"ETF {i+1}", key=f"entry_{i}", value=val)
    with col2:
        st.button("Remove", key=f"del_{i}", on_click=remove_entry, args=(i,))

for i in range(len(st.session_state.entries)):
    st.session_state.entries[i] = st.session_state[f"entry_{i}"]

col1, col2 = st.columns([1, 6])
with col1:
    st.button("Add One More", on_click=add_entry)
with col2:
    st.button("Fetch Data", on_click=calculate)

st.markdown("---")

# Upload price data
uploaded_file = st.file_uploader("Upload ETF daily price CSV (same format as Part 1)", type=["csv"])

# Questionnaire inputs
allow_short = st.toggle("Allow Short Sales in Optimal Portfolio?", value=True)

st.header(" Investor Questionnaire")
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
if (uploaded_file or st.session_state.use_input) and st.button("Calculate Optimal Portfolio") :
    # Load and clean data
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
        df = df.dropna()
        st.success("File uploaded successfully.")
    else:
        df = st.session_state.calc_result
        st.success("Data fetched successfully.")

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
    bounds = [(-2 / n_assets, 2 / n_assets)] * n_assets if allow_short else [(0, 2 / n_assets)] * n_assets

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

    # ---- Optimal Portfolio Output ----
    st.markdown("---")
    st.markdown("### Optimal Portfolio Based on Your Risk Aversion")

    # Prepare weights table
    weights_df = pd.DataFrame({
        'Fund': mean_returns.index,
        'Weight': optimal_weights.flatten()
    }).set_index("Fund")

    # Split layout: table + pie chart
    col1, col2 = st.columns([1, 1.2])

    # ---- Column 1: Weights Table ----
    with col1:
        st.dataframe(weights_df.style.format("{:.2%}"), use_container_width=True)

    # ---- Column 2: Interactive Pie Chart ----
    with col2:
        if (weights_df["Weight"] < 0).any():
            st.warning("Your optimal portfolio includes short positions. Negative weights are excluded from the pie chart.")

        # Only non-negative and meaningful weights
        weights_nonneg = weights_df.copy()
        weights_nonneg["Weight"] = weights_nonneg["Weight"].clip(lower=0)
        plot_df = weights_nonneg[weights_nonneg["Weight"] > 0.01]

        if plot_df.empty:
            st.warning("No positive weights above 1% to display in pie chart.")
        else:
            fig_pie = px.pie(
                plot_df.reset_index(),
                names="Fund",
                values="Weight",
                title="Portfolio Allocation (Non-Negative Weights Only)",
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig_pie.update_traces(textinfo='percent+label', pull=[0.03]*len(plot_df))
            fig_pie.update_layout(
                height=400,
                margin=dict(t=50, b=40, l=40, r=40),
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(f"""
    - **Expected Return**: {port_return:.2%}  
    - **Portfolio Volatility**: {np.sqrt(port_var):.2%}  
    - **Utility Score (U)**: {utility:.4f}
    """)

    # Define a range of A values draw the graph
    A_values = np.linspace(0.1, 10, 100)  # Risk aversion from 0.1 to 10
    U_values = [port_return - (A * port_var / 2) for A in A_values]

    # Build DataFrame for plotting
    utility_df = pd.DataFrame({
        "Risk Aversion (A)": A_values,
        "Utility Score (U)": U_values
    })

    # Create Plotly line chart
    fig_utility = px.line(
        utility_df,
        x="Risk Aversion (A)",
        y="Utility Score (U)",
        title="Utility Score vs. Risk Aversion Level",
        markers=True
    )

    fig_utility.update_layout(
        xaxis_title="Risk Aversion Coefficient (A)",
        yaxis_title="Utility Score (U)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Display the plot
    st.plotly_chart(fig_utility, use_container_width=True)
    # Portfolio Return vs Volatility (σ), fixed A
    # -- Volatility range
    sigma_range = np.linspace(0.01, 0.5, 100)

    # -- Live interactive UI block
    with st.container():
        risk_aversion = A

        # Recalculate U for each sigma
        U_curve = [port_return - (risk_aversion * sigma**2 / 2) for sigma in sigma_range]

        fig_risk = px.line(
            x=sigma_range,
            y=U_curve,
            labels={"x": "Portfolio Volatility (σ)", "y": "Utility Score (U)"},
            title=f"Utility vs. Volatility (A = {risk_aversion:.1f})"
        )

        fig_risk.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig_risk, use_container_width=True)

    
    # 1. Efficient Frontier with Utility Contours (Contour Plot)
    # Simulated grid for standard deviation (σ) and return (r)
    sigma_vals = np.linspace(0.01, 0.4, 100)
    return_vals = np.linspace(0, 0.2, 100)
    Sigma, R = np.meshgrid(sigma_vals, return_vals)

    # Utility surface for current A
    U_grid = R - (risk_aversion * Sigma**2) / 2

    # Create contour plot
    fig_contour = go.Figure(data=go.Contour(
        z=U_grid,
        x=sigma_vals,
        y=return_vals,
        colorscale='Viridis',
        contours=dict(showlabels=True),
        colorbar=dict(title='Utility (U)'),
    ))

    fig_contour.update_layout(
        title=f"Utility Contour Map (A = {risk_aversion})",
        xaxis_title="Volatility (σ)",
        yaxis_title="Expected Return (r)",
        height=500,
        template="plotly_white"
    )

    st.plotly_chart(fig_contour, use_container_width=True)
    #2. Risk-Return Scatter Plot with Utility-Based Sizing

    # Compute utility for each portfolio
    results_short = simulate_portfolios(mean_returns, cov_matrix, allow_short=True)
    utility_all = results_short[1] - (risk_aversion * results_short[0]**2 / 2)

    df_bubble = pd.DataFrame({
        "Risk": results_short[0],
        "Return": results_short[1],
        "Sharpe": results_short[2],
        "Utility": utility_all
    })

    df_bubble["Utility Size"] = df_bubble["Utility"] - df_bubble["Utility"].min()
    df_bubble["Utility Size"] /= df_bubble["Utility Size"].max()
    df_bubble["Utility Size"] = df_bubble["Utility Size"].clip(lower=0.05) * 40  # scale up

    fig_bubble = px.scatter(
        df_bubble,
        x="Risk",
        y="Return",
        size="Utility Size",
        color="Sharpe",
        color_continuous_scale="Viridis",
        title=" Normalized Utility Sizing of Portfolios",
        labels={"Sharpe": "Sharpe Ratio"}
    )

    fig_bubble.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_bubble, use_container_width=True)

    #3. Risk Aversion Slider Trace (Live Trace of Utility vs A)
    A_range = np.linspace(0.1, 10, 100)
    U_trace = [port_return - (A * port_var / 2) for A in A_range]

    fig_trace = px.line(
        x=A_range,
        y=U_trace,
        labels={"x": "Risk Aversion (A)", "y": "Utility (U)"},
        title="Utility Score vs. Risk Aversion for Fixed Portfolio"
    )

    fig_trace.update_traces(mode="lines+markers")
    fig_trace.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_trace, use_container_width=True)








# Always show history at the bottom
if st.session_state.get("history"):
    st.markdown("---")
    st.markdown("### Session History of Calculations")

    # Convert session history to DataFrame
    history_df = pd.DataFrame(st.session_state.history)

    # Display styled table
    col1, col2, col3 = st.columns([0.2, 1.6, 0.2])
    with col2:
        st.dataframe(
            history_df.style.format({
                "Expected Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Utility": "{:.4f}"
            }),
            use_container_width=True
        )

    # Optional: plot only if 2+ data points
    if len(history_df) > 1:
        st.markdown("### Utility and Risk Evolution Over Time")

        # Add step count or timestamp
        history_df["Step"] = range(1, len(history_df)+1)

        # Create interactive line plot
        fig_history = px.line(
            history_df,
            x="Step",
            y=["Expected Return", "Volatility", "Utility"],
            markers=True,
            title="Metric Trends Across User Inputs",
            labels={"value": "Metric Value", "Step": "Interaction Step", "variable": "Metric"},
        )

        fig_history.update_layout(
            template="plotly_white",
            height=500,
            legend_title="Metric",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig_history, use_container_width=True)
