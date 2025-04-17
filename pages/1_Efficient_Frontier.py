import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.set_page_config(page_title="Bond Portfolio Analyzer", layout="wide")
st.title("Academic Bond Portfolio Analyzer")

st.markdown(
    """
This interactive platform lets you analyze bond ETF portfolios using portfolio theory.

Upload a **daily price CSV file** with a `Date` column and bond prices as other columns.
"""
)

uploaded_file = st.file_uploader("Upload your bond price CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.dropna()
    st.success("File uploaded successfully.")

    # ---- Data Preview ----
    st.markdown("### Uploaded Bond Price Data")

    # Centered layout using Streamlit columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(df.head(), use_container_width=True)
    # ---- Historical Bond Price Chart ----
    st.markdown("### Bond Price Trends Over Time")

    # Reset index for plotting (Date is currently the index)
    df_plot = df.reset_index().melt(id_vars="Date", var_name="Bond", value_name="Price")

    # Create interactive time series chart
    fig_price_trend = px.line(
        df_plot,
        x="Date",
        y="Price",
        color="Bond",
        title="Historical Bond ETF Prices",
        labels={"Price": "Daily Price", "Date": "Date", "Bond": "Bond Fund"},
    )

    fig_price_trend.update_layout(
        template="plotly_white",
        height=500,
        legend_title="Bond Fund",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Display chart
    st.plotly_chart(fig_price_trend, use_container_width=True)
    # Calculate returns
    returns = df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    std_devs = returns.std() * np.sqrt(252)
    sharpe_ratios = mean_returns / std_devs

    # ---- Descriptive Summary Statistics ----
    st.markdown("### Descriptive Statistics")

    # Create and format summary table
    stats_df = pd.DataFrame({
        "Annualized Return": mean_returns,
        "Annualized Volatility": std_devs,
        "Sharpe Ratio": sharpe_ratios
    })

    # Center display using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(
            stats_df.style.format("{:.4f}").set_caption("Portfolio Summary Statistics"),
            use_container_width=True
        )
    # ---- Descriptive Statistics Chart (Interactive) ----
    st.markdown("### Visual Summary of Portfolio Statistics")

    # Prepare data for plotting
    stats_df_reset = stats_df.reset_index().rename(columns={"index": "Fund"})
    stats_melted = stats_df_reset.melt(id_vars="Fund", var_name="Metric", value_name="Value")

    # Plot grouped bar chart
    fig_stats = px.bar(
        stats_melted,
        x="Fund",
        y="Value",
        color="Metric",
        barmode="group",
        text_auto=".4f",
        labels={"Value": "Metric Value"},
        title="Descriptive Statistics by Fund"
    )

    fig_stats.update_layout(
        xaxis_title="Bond Fund",
        yaxis_title="Value",
        legend_title="Metric",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Show plot in Streamlit
    st.plotly_chart(fig_stats, use_container_width=True)

    # ---- Cumulative Returns ----
    st.subheader("Cumulative Returns")
    cumulative = (1 + returns).cumprod()
    st.line_chart(cumulative)

    # ---- Correlation Heatmap ----
    st.subheader("Correlation Matrix")
    corr = returns.corr()
    # ---- Correlation Matrix Table ----
    st.markdown("### Correlation Matrix (Table View)")

    # Center and format correlation table
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    # ---- Correlation Heatmap (Interactive Plotly Version) ----
    st.markdown("### Interactive Correlation Heatmap")

    # Convert correlation matrix to numpy for plotly
    z = corr.values
    x_labels = corr.columns.tolist()
    y_labels = corr.index.tolist()

    # Create the heatmap using Plotly
    fig_corr_plotly = ff.create_annotated_heatmap(
        z,
        x=x_labels,
        y=y_labels,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        annotation_text=[[f"{val:.2f}" for val in row] for row in z],
        showscale=True,
        hoverinfo="z"
    )

    fig_corr_plotly.update_layout(
        title_text="Bond ETF Correlation Matrix (Interactive)",
        xaxis=dict(side="bottom"),
        width=800,
        height=600,
        margin=dict(l=80, r=40, t=60, b=80)
    )

    st.plotly_chart(fig_corr_plotly, use_container_width=True)

    # ---- Daily Return Samples ----
    st.markdown("### Sample of Daily Returns")
    st.dataframe(returns.head().style.format("{:.4f}"), use_container_width=True)
    # ---- Distribution of Daily Returns (Interactive Plotly Version) ----
    st.markdown("### 🧪 Interactive Distribution of Daily Returns")

    # Melt the returns DataFrame for long-form plotting
    returns_long = returns.reset_index().melt(id_vars="Date", var_name="Fund", value_name="Daily Return")

    # Create KDE-style density plot using histogram with density normalization
    fig_kde_plotly = px.histogram(
        returns_long,
        x="Daily Return",
        color="Fund",
        marginal="rug",  # optional: add small tick marks
        opacity=0.5,
        nbins=100,
        histnorm="density",
        barmode="overlay",
        title="Distribution of Daily Returns (Interactive View)"
    )

    fig_kde_plotly.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Daily Return",
        yaxis_title="Density",
        legend_title="Bond Fund",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Show plot in Streamlit
    st.plotly_chart(fig_kde_plotly, use_container_width=True)
    # ---- Portfolio Simulation Functions ----
    def simulate_portfolios(
        mean_returns, cov_matrix, n_portfolios=5000, allow_short=False
    ):
        np.random.seed(42)
        n_assets = len(mean_returns)
        results = np.zeros((3, n_portfolios))

        for i in range(n_portfolios):
            weights = (
                np.random.randn(n_assets) if allow_short else np.random.rand(n_assets)
            )
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
        df = pd.DataFrame({"std": results[0], "ret": results[1]}).sort_values(by="std")
        return df[df["ret"] == df["ret"].cummax()]

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

    # ---- Frontier WITH Short Sales (Interactive Plotly Version) ----
    st.markdown("### Efficient Frontier (With Short Sales)")

    # Extract efficient frontier and CML
    frontier_short = extract_efficient_frontier(results_short)
    tangency_point_short, cml_x_short, cml_y_short = get_tangency_line(results_short)

    # Create interactive figure
    fig1 = go.Figure()

    # Simulated Portfolios
    fig1.add_trace(go.Scatter(
        x=results_short[0],
        y=results_short[1],
        mode='markers',
        marker=dict(
            color=results_short[2],
            colorscale='Viridis',
            size=6,
            opacity=0.4,
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Simulated Portfolios',
        hovertemplate='Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>'
    ))

    # Efficient Frontier
    fig1.add_trace(go.Scatter(
        x=frontier_short["std"],
        y=frontier_short["ret"],
        mode='lines',
        line=dict(color='black', width=3),
        name='Efficient Frontier'
    ))

    # Capital Market Line (CML)
    fig1.add_trace(go.Scatter(
        x=cml_x_short,
        y=cml_y_short,
        mode='lines',
        line=dict(dash='dash', color='blue'),
        name='Capital Market Line'
    ))

    # Tangency Portfolio
    fig1.add_trace(go.Scatter(
        x=[tangency_point_short[0]],
        y=[tangency_point_short[1]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='circle'),
        name='Tangency Portfolio'
    ))

    # Individual Funds
    fig1.add_trace(go.Scatter(
        x=np.sqrt(np.diag(cov_matrix)),
        y=mean_returns,
        mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name='Individual Funds'
    ))

    # GMVP
    fig1.add_trace(go.Scatter(
        x=[gmvp_std],
        y=[gmvp_return],
        mode='markers',
        marker=dict(color='gold', size=16, symbol='star', line=dict(color='black', width=1)),
        name='GMVP'
    ))

    # Final layout
    fig1.update_layout(
        title='Efficient Frontier (With Short Sales)',
        xaxis_title='Risk (Standard Deviation)',
        yaxis_title='Expected Return',
        template='plotly_white',
        height=600,
        hovermode='closest',
        legend_title='Legend',
        margin=dict(l=60, r=40, t=60, b=60)
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig1, use_container_width=True)

   
    # ---- Show top 10 efficient frontier points (centered) ----
    st.markdown("###  Efficient Frontier Data (With Short Sales)")

    # Use columns to center the dataframe
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(
            frontier_short.head(10).style.format({"std": "{:.4f}", "ret": "{:.4f}"}),
            use_container_width=True
        )
    # ---- Frontier WITHOUT Short Sales (Interactive) ----
    st.markdown("### Efficient Frontier (Without Short Sales)")

    # Extract data
    frontier_noshort = extract_efficient_frontier(results_noshort)
    tangency_point_noshort, cml_x_noshort, cml_y_noshort = get_tangency_line(results_noshort)

    # Create Plotly figure
    fig2 = go.Figure()

    # Simulated portfolios
    fig2.add_trace(go.Scatter(
        x=results_noshort[0],
        y=results_noshort[1],
        mode='markers',
        marker=dict(
            color=results_noshort[2],
            colorscale='Plasma',
            size=6,
            opacity=0.4,
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Simulated Portfolios',
        hovertemplate='Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>'
    ))

    # Efficient frontier
    fig2.add_trace(go.Scatter(
        x=frontier_noshort["std"],
        y=frontier_noshort["ret"],
        mode='lines',
        line=dict(color='black', width=3),
        name='Efficient Frontier'
    ))

    # Capital Market Line
    fig2.add_trace(go.Scatter(
        x=cml_x_noshort,
        y=cml_y_noshort,
        mode='lines',
        line=dict(dash='dash', color='blue'),
        name='Capital Market Line'
    ))

    # Tangency Portfolio
    fig2.add_trace(go.Scatter(
        x=[tangency_point_noshort[0]],
        y=[tangency_point_noshort[1]],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='circle'),
        name='Tangency Portfolio'
    ))

    # Individual funds
    fig2.add_trace(go.Scatter(
        x=np.sqrt(np.diag(cov_matrix)),
        y=mean_returns,
        mode='markers',
        marker=dict(color='red', symbol='x', size=10),
        name='Individual Funds'
    ))

    # GMVP
    fig2.add_trace(go.Scatter(
        x=[gmvp_std],
        y=[gmvp_return],
        mode='markers',
        marker=dict(color='gold', symbol='star', size=16, line=dict(color='black', width=1)),
        name='GMVP'
    ))

    # Layout and styling
    fig2.update_layout(
        title='Efficient Frontier (Without Short Sales)',
        xaxis_title='Risk (Standard Deviation)',
        yaxis_title='Expected Return',
        legend_title='Legend',
        template='plotly_white',
        height=600,
        hovermode='closest',
        margin=dict(l=60, r=40, t=60, b=60)
    )

    # Display in Streamlit
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Show top 10 efficient frontier points WITHOUT short sales (centered) ----
    st.markdown("###  Efficient Frontier Data (Without Short Sales)")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(
            frontier_noshort.head(10).style.format({"std": "{:.4f}", "ret": "{:.4f}"}),
            use_container_width=True
        )

    # ---- GMVP Weights (centered) ----
    st.markdown("### Global Minimum Variance Portfolio (GMVP) Weights")

    col4, col5, col6 = st.columns([1, 2, 1])
    with col5:
        gmvp_df = pd.DataFrame({
            "Fund": mean_returns.index,
            "GMVP Weight": gmvp_weights
        })
        st.dataframe(
            gmvp_df.set_index("Fund").style.format("{:.4%}"),
            use_container_width=True
        )
    # ---- GMVP Weights Chart (Advanced) ----
    # 🧮 Convert and sort GMVP weights
    gmvp_df["GMVP Weight (%)"] = gmvp_df["GMVP Weight"] * 100
    gmvp_df_sorted = gmvp_df.sort_values(by="GMVP Weight (%)", ascending=True)

    #  Plot GMVP weight distribution (horizontal bar chart)
    fig_weights = px.bar(
        gmvp_df_sorted,
        x="GMVP Weight (%)",
        y="Fund",
        orientation="h",
        text="GMVP Weight (%)",
        color="GMVP Weight (%)",
        color_continuous_scale="Blues",
        labels={"GMVP Weight (%)": "Weight (%)", "Fund": "Bond Fund"},
        title="Global Minimum Variance Portfolio (GMVP) Weight Distribution"
    )

    fig_weights.update_traces(
        texttemplate='%{text:.2f}%', 
        textposition='outside'
    )

    fig_weights.update_layout(
        xaxis_title="Weight (%)",
        yaxis_title="",
        coloraxis_showscale=False,
        height=500,
        margin=dict(l=100, r=40, t=60, b=40)
    )

    st.plotly_chart(fig_weights, use_container_width=True)
# ---- Academic Formulas (LaTeX Block) ----
st.markdown(
    r"""
---
### Academic Concepts Summary

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
"""
)
