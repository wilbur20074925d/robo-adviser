import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bond Portfolio Analyzer", layout="wide")
st.title("Academic Bond Portfolio Analyzer")

st.markdown("""
This interactive platform lets you analyze bond ETF portfolios using portfolio theory.

Upload a **daily price CSV file** with a `Date` column and bond prices as other columns.
""")

uploaded_file = st.file_uploader("Upload your bond price CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.dropna()
    st.success("File uploaded successfully.")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Calculate returns
    returns = df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    std_devs = returns.std() * np.sqrt(252)
    sharpe_ratios = mean_returns / std_devs

    # ---- Summary Statistics ----
    st.subheader("Descriptive Statistics")
    stats_df = pd.DataFrame({
        "Annualized Return": mean_returns,
        "Annualized Volatility": std_devs,
        "Sharpe Ratio": sharpe_ratios
    })
    st.dataframe(stats_df.style.format("{:.4f}"))

    # ---- Cumulative Returns ----
    st.subheader("Cumulative Returns")
    cumulative = (1 + returns).cumprod()
    st.line_chart(cumulative)

    # ---- Correlation Heatmap ----
    st.subheader("Correlation Matrix")
    corr = returns.corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

    # ---- Return Distributions ----
    st.subheader("Distribution of Daily Returns")
    fig_hist, ax = plt.subplots(figsize=(12, 6))
    for col in returns.columns:
        sns.kdeplot(returns[col], label=col, ax=ax)
    ax.legend()
    ax.set_title("Distribution of Daily Returns")
    st.pyplot(fig_hist)

    # ---- Portfolio Simulation Functions ----
    def simulate_portfolios(mean_returns, cov_matrix, n_portfolios=5000, allow_short=False):
        np.random.seed(42)
        n_assets = len(mean_returns)
        results = np.zeros((3, n_portfolios))

        for i in range(n_portfolios):
            weights = np.random.randn(n_assets) if allow_short else np.random.rand(n_assets)
            weights /= np.sum(np.abs(weights)) if allow_short else np.sum(weights)
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(weights.T @ cov_matrix @ weights)
            results[0, i] = port_std
            results[1, i] = port_return
            results[2, i] = (port_return - 0.02) / port_std  # Sharpe with rf = 2%
        return results

    def calculate_gmvp(cov_matrix, mean_returns):
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        port_return = weights.T @ mean_returns
        port_std = np.sqrt(weights.T @ cov_matrix @ weights)
        return weights, port_return, port_std

    def extract_efficient_frontier(results):
        df = pd.DataFrame({
            "std": results[0],
            "ret": results[1]
        }).sort_values(by="std")
        return df[df['ret'] == df['ret'].cummax()]

    def get_tangency_line(results, risk_free_rate=0.02):
        stds, rets = results[0], results[1]
        sharpe_ratios = (rets - risk_free_rate) / stds
        max_idx = np.argmax(sharpe_ratios)
        max_sharpe_std = stds[max_idx]
        max_sharpe_ret = rets[max_idx]
        max_sharpe = sharpe_ratios[max_idx]
        cml_x = np.linspace(0, stds.max(), 100)
        cml_y = risk_free_rate + max_sharpe * cml_x
        return (max_sharpe_std, max_sharpe_ret, max_sharpe), cml_x, cml_y

    # ---- Simulations ----
    results_short = simulate_portfolios(mean_returns, cov_matrix, allow_short=True)
    results_noshort = simulate_portfolios(mean_returns, cov_matrix, allow_short=False)
    gmvp_weights, gmvp_return, gmvp_std = calculate_gmvp(cov_matrix, mean_returns)

    # ---- Frontier WITH Short Sales ----
    st.subheader("Efficient Frontier (With Short Sales)")
    frontier_short = extract_efficient_frontier(results_short)
    tangency_point_short, cml_x_short, cml_y_short = get_tangency_line(results_short)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(results_short[0], results_short[1], c=results_short[2], cmap='viridis', alpha=0.3, label='Portfolios')
    ax1.plot(frontier_short["std"], frontier_short["ret"], color='black', linewidth=2.5, label='Efficient Frontier')
    ax1.plot(cml_x_short, cml_y_short, linestyle='--', color='blue', label='Capital Market Line')
    ax1.scatter(tangency_point_short[0], tangency_point_short[1], c='blue', marker='o', s=120, label='Tangency Portfolio')
    ax1.scatter(np.sqrt(np.diag(cov_matrix)), mean_returns, c='red', marker='x', s=100, label='Individual Funds')
    ax1.scatter(gmvp_std, gmvp_return, c='gold', edgecolors='black', marker='*', s=250, label='GMVP')
    ax1.set_title('Efficient Frontier with Short Sales')
    ax1.set_xlabel('Risk (Standard Deviation)')
    ax1.set_ylabel('Expected Return')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    st.markdown("#### Efficient Frontier Data (With Short Sales)")
    st.dataframe(frontier_short.head(10).style.format({'std': "{:.4f}", 'ret': "{:.4f}"}))

    # ---- Frontier WITHOUT Short Sales ----
    st.subheader("Efficient Frontier (Without Short Sales)")
    frontier_noshort = extract_efficient_frontier(results_noshort)
    tangency_point_noshort, cml_x_noshort, cml_y_noshort = get_tangency_line(results_noshort)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(results_noshort[0], results_noshort[1], c=results_noshort[2], cmap='plasma', alpha=0.3, label='Portfolios')
    ax2.plot(frontier_noshort["std"], frontier_noshort["ret"], color='black', linewidth=2.5, label='Efficient Frontier')
    ax2.plot(cml_x_noshort, cml_y_noshort, linestyle='--', color='blue', label='Capital Market Line')
    ax2.scatter(tangency_point_noshort[0], tangency_point_noshort[1], c='blue', marker='o', s=120, label='Tangency Portfolio')
    ax2.scatter(np.sqrt(np.diag(cov_matrix)), mean_returns, c='red', marker='x', s=100, label='Individual Funds')
    ax2.scatter(gmvp_std, gmvp_return, c='gold', edgecolors='black', marker='*', s=250, label='GMVP')
    ax2.set_title('Efficient Frontier without Short Sales')
    ax2.set_xlabel('Risk (Standard Deviation)')
    ax2.set_ylabel('Expected Return')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("#### Efficient Frontier Data (Without Short Sales)")
    st.dataframe(frontier_noshort.head(10).style.format({'std': "{:.4f}", 'ret': "{:.4f}"}))

    # ---- GMVP Weights ----
    st.subheader("Global Minimum Variance Portfolio (GMVP) Weights")
    gmvp_df = pd.DataFrame({
        "Fund": mean_returns.index,
        "GMVP Weight": gmvp_weights
    })
    st.dataframe(gmvp_df.set_index("Fund").style.format("{:.4%}"))

# ---- Academic Formulas (LaTeX Block) ----
st.markdown(r"""
---
### 🧠 Academic Concepts Summary

**1. Annualized Return ($\mu$):**  
$$
\mu = \text{mean(daily return)} \times 252
$$

**2. Annualized Volatility ($\sigma$):**  
$$
\sigma = \text{std(daily return)} \times \sqrt{252}
$$

**3. Sharpe Ratio (w.r.t. risk-free rate $r_f = 0.02$):**  
$$
\text{Sharpe Ratio} = \frac{\mu - r_f}{\sigma}
$$

**4. Global Minimum Variance Portfolio (GMVP):**  
$$
w = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}
$$

Where $\Sigma$ is the covariance matrix of asset returns.
""")