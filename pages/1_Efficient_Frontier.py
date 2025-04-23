import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import cvxpy as cp

st.set_page_config(page_title="ETF Portfolio Analyzer", layout="wide")
st.title("Academic Portfolio Analyzer")

st.markdown(
    """
This interactive platform lets you analyze ETF portfolios using portfolio theory

Upload a **daily price CSV file** with a `Date` column and ETF prices as other columns.
"""
)

uploaded_file = st.file_uploader("Upload your price CSV", type=["csv"])
risk_free_rate = st.number_input(
    "Please input risk-free rate (default set as 2%)",
    min_value=0.0,
    max_value=100.0,
    value=2.0,
    step=0.1
) / 100

if uploaded_file and risk_free_rate:
    # Load data
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.dropna()
    st.success("File uploaded successfully.")

    # ---- Data Preview ----
    st.markdown("### Uploaded ETF Price Data (First Five Rows)")

    # Centered layout using Streamlit columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(df.head(), use_container_width=True)
    # ---- Historical Price Chart ----
    st.markdown("### Price Trends Over Time")

    # Reset index for plotting (Date is currently the index)
    df_plot = df.reset_index().melt(id_vars="Date", var_name="ETF", value_name="Price")

    # Create interactive time series chart
    fig_price_trend = px.line(
        df_plot,
        x="Date",
        y="Price",
        color="ETF",
        title="Historical ETF Prices",
        labels={"Price": "Daily Price", "Date": "Date", "ETF": "ETF"},
    )

    fig_price_trend.update_layout(
        template="plotly_white",
        height=500,
        legend_title="ETF",
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
    sharpe_ratios = (mean_returns - risk_free_rate) / std_devs

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
    
    # ---- Covariance Matrix Table ----
    st.markdown("### Covariance Matrix (Annualized)")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(cov_matrix.style.format("{:.6f}"), use_container_width=True)

    # ---- Correlation Heatmap ----
    st.subheader("Correlation Matrix")
    corr = returns.corr()
    # ---- Correlation Matrix Table ----
    st.markdown("### Correlation Matrix (Table View)")

    # Center and format correlation table
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    # ---- Correlation Heatmap (Plotly Version) ----
    st.markdown("### Correlation Heatmap")

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
        title_text="ETF Correlation Matrix (Interactive)",
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
    
    # ---- Portfolio Simulation Functions ----
    def simulate_portfolios(
        mean_returns, cov_matrix, risk_free_rate, n_portfolios=10000, allow_short=False, 
    ):
        np.random.seed(42)
        n_assets = len(mean_returns)
        results = np.zeros((3, n_portfolios))
        weights_array = np.zeros((n_portfolios, n_assets))
        for i in range(n_portfolios):
            weights = (
                np.random.randn(n_assets) if allow_short else np.random.rand(n_assets)
            )
            weights /= np.sum(np.abs(weights)) if allow_short else np.sum(weights)
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(weights.T @ cov_matrix @ weights)
            results[0, i] = port_std
            results[1, i] = port_return
            results[2, i] = (port_return - risk_free_rate) / port_std
            weights_array[i, :] = weights
        return results, weights_array

    def calculate_gmvp(cov_matrix, mean_returns):
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        port_return = weights.T @ mean_returns
        port_std = np.sqrt(weights.T @ cov_matrix @ weights)
        return weights, port_return, port_std
    
    def calculate_gmvp_noshort(mean_returns, cov_matrix):
    
        n = len(mean_returns)  
        w_longonly = cp.Variable(n)  

        # Target: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(w_longonly, cov_matrix))  

        # Contraint: 1. weight sum equal to 1, 2. weight >= 0
        constraints = [  
            cp.sum(w_longonly) == 1,  
            w_longonly >= 0  
        ]  

        prob = cp.Problem(objective, constraints)  
        prob.solve()  

        weights_longonly = w_longonly.value 
        return_longonly  = weights_longonly @ mean_returns
        std_longonly     = np.sqrt(weights_longonly.T @ cov_matrix @ weights_longonly)
        
        return weights_longonly, return_longonly, std_longonly

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
        return (max_sharpe_std, max_sharpe_ret, max_sharpe), max_idx, cml_x, cml_y

    # ---- Simulations ----
    results_short, weights_short = simulate_portfolios(mean_returns, cov_matrix, risk_free_rate, allow_short=True)
    results_noshort, weights_noshort = simulate_portfolios(mean_returns, cov_matrix, risk_free_rate, allow_short=False)
    gmvp_weights, gmvp_return, gmvp_std = calculate_gmvp(cov_matrix, mean_returns)
    gmvp_weights_noshort, gmvp_return_noshort, gmvp_std_noshort = calculate_gmvp_noshort(mean_returns, cov_matrix)
    

    # ---- Frontier WITH Short Sales (Interactive Plotly Version) ----
    st.markdown("### Efficient Frontier (With Short Sales)")

    # Extract efficient frontier and CML
    frontier_short = extract_efficient_frontier(results_short)
    tangency_point_short, max_idx_short, cml_x_short, cml_y_short = get_tangency_line(results_short, risk_free_rate)

    # Create interactive figure
    fig1 = go.Figure()
    
    def get_weight_label(weight):
        "<br>".join(
            f"{name}: {val}" for name,val in zip(mean_returns.index, [f"{pct:.2%}" for pct in weight])
        )
    
    labels_short = []
    for w in weights_short:
        labels_short.append("<br>".join(
            f"{name}: {val}" for name,val in zip(mean_returns.index, [f"{pct:.2%}" for pct in w])
        ))
    labels_noshort = []
    for w in weights_noshort:
        labels_noshort.append("<br>".join(
            f"{name}: {val}" for name,val in zip(mean_returns.index, [f"{pct:.2%}" for pct in w])
        ))
    gmvp_short_label = ["<br>".join(
            f"{name}: {val}" for name,val in zip(mean_returns.index, [f"{pct:.2%}" for pct in gmvp_weights])
        )]
    gmvp_noshort_label = ["<br>".join(
            f"{name}: {val}" for name,val in zip(mean_returns.index, [f"{pct:.2%}" for pct in gmvp_weights_noshort])
        )]

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
        customdata=labels_short,
        hovertemplate=(
        "Risk: %{x:.4f}<br>"
        "Return: %{y:.4f}<br>"
        "<b>Weights:</b><br>%{customdata}<extra></extra>"
    )
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
        name='Tangency Portfolio',
        customdata=[labels_short[max_idx_short]],
        hovertemplate=(
        "<b>Tangency Portfolio:</b> <br>"
        "Risk: %{x:.4f}<br>"
        "Return: %{y:.4f}<br>"
        "<b>Weights:</b><br>%{customdata}<extra></extra>"
    )
    ))

    # Individual Funds
    etf_names = mean_returns.index.tolist()
    fig1.add_trace(go.Scatter(
        x=np.sqrt(np.diag(cov_matrix)),
        y=mean_returns,
        mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name='Individual Funds',
        customdata=np.array(etf_names).reshape(-1, 1),
        hovertemplate='ETF: %{customdata[0]}<br>Return: %{y:.4%}<br>Risk: %{x:.4%}<extra></extra>'
    ))

    # GMVP
    fig1.add_trace(go.Scatter(
        x=[gmvp_std],
        y=[gmvp_return],
        mode='markers',
        marker=dict(color='gold', size=16, symbol='star', line=dict(color='black', width=1)),
        name='GMVP',
        customdata=gmvp_short_label,
        hovertemplate=(
        "<b>GMVP:</b> <br>"
        "Risk: %{x:.4f}<br>"
        "Return: %{y:.4f}<br>"
        "<b>Weights:</b><br>%{customdata}<extra></extra>"
    )))

    # Final layout
    fig1.update_layout(
        title='Efficient Frontier (With Short Sales)',
        xaxis_title='Risk (Standard Deviation)',
        yaxis_title='Expected Return',
        template='plotly_white',
        height=600,
        hovermode='closest',
        legend_title='Legend',
        margin=dict(l=60, r=60, t=60, b=60)
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Frontier WITHOUT Short Sales (Interactive) ----
    st.markdown("### Efficient Frontier (Without Short Sales)")

    # Extract data
    frontier_noshort = extract_efficient_frontier(results_noshort)
    tangency_point_noshort, max_idx_noshort, cml_x_noshort, cml_y_noshort = get_tangency_line(results_noshort, risk_free_rate)

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
        customdata=labels_noshort,
        hovertemplate=(
        "Risk: %{x:.4f}<br>"
        "Return: %{y:.4f}<br>"
        "<b>Weights:</b><br>%{customdata}<extra></extra>"
    )
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
        name='Tangency Portfolio',
        customdata=[labels_noshort[max_idx_noshort]],
        hovertemplate=(
        "<b>Tangency Portfolio:</b> <br>"
        "Risk: %{x:.4f}<br>"
        "Return: %{y:.4f}<br>"
        "<b>Weights:</b><br>%{customdata}<extra></extra>"
    )
    ))

    # Individual funds
    fig2.add_trace(go.Scatter(
        x=np.sqrt(np.diag(cov_matrix)),
        y=mean_returns,
        mode='markers',
        marker=dict(color='red', symbol='x', size=10),
        name='Individual Funds',
        customdata=np.array(etf_names).reshape(-1, 1),
        hovertemplate='ETF: %{customdata[0]}<br>Return: %{y:.4%}<br>Risk: %{x:.4%}<extra></extra>'
    ))

    # GMVP
    fig2.add_trace(go.Scatter(
        x=[gmvp_std_noshort],
        y=[gmvp_return_noshort],
        mode='markers',
        marker=dict(color='gold', symbol='star', size=16, line=dict(color='black', width=1)),
        name='GMVP',
        customdata=gmvp_noshort_label,
        hovertemplate=(
        "<b>GMVP:</b> <br>"
        "Risk: %{x:.4f}<br>"
        "Return: %{y:.4f}<br>"
        "<b>Weights:</b><br>%{customdata}<extra></extra>"
    )))

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
    
    # ---- Tangency Portfolio ----
    st.markdown("### Tangency Portfolio Weights")

    print(mean_returns.index)
    print(weights_short[max_idx_short])
    print(weights_noshort[max_idx_noshort])
    col4, col5, col6 = st.columns([1, 2, 1])
    with col5:
        tan_df = pd.DataFrame({
            "Fund": mean_returns.index,
            "Tangency Portfolio Weight (With short)": weights_short[max_idx_short],
            "Tangency Portfolio Weight (Without short)": weights_noshort[max_idx_noshort]
        })
        st.dataframe(
            tan_df.set_index("Fund").style.format("{:.4%}"),
            use_container_width=True
        )
    # ---- Tangency Portfolio Weights Chart (Advanced) ----
    # Convert and sort Tangency Portfolio weights
    tan_df["Tangency Portfolio Weight With Short (%)"] = tan_df["Tangency Portfolio Weight (With short)"] * 100
    tan_df_sorted = tan_df.sort_values(by="Tangency Portfolio Weight With Short (%)", ascending=True)

    #  Plot Tangency Portfolio weight distribution (horizontal bar chart)
    fig_weights = px.bar(
        tan_df_sorted,
        x="Tangency Portfolio Weight With Short (%)",
        y="Fund",
        orientation="h",
        text="Tangency Portfolio Weight With Short (%)",
        color="Tangency Portfolio Weight With Short (%)",
        color_continuous_scale="Blues",
        labels={"Tangency Portfolio Weight With Short (%)": "Weight (%)", "Fund": "ETF"},
        title="Tangency Portfolio Weight Distribution With Short"
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
    
    
    # Convert and sort Tangency Portfolio weights
    tan_df["Tangency Portfolio Weight Without Short (%)"] = tan_df["Tangency Portfolio Weight (Without short)"] * 100
    tan_df_sorted = tan_df.sort_values(by="Tangency Portfolio Weight Without Short (%)", ascending=True)

    #  Plot Tangency Portfolio weight distribution (horizontal bar chart)
    fig_weights = px.bar(
        tan_df_sorted,
        x="Tangency Portfolio Weight Without Short (%)",
        y="Fund",
        orientation="h",
        text="Tangency Portfolio Weight Without Short (%)",
        color="Tangency Portfolio Weight Without Short (%)",
        color_continuous_scale="Blues",
        labels={"Tangency Portfolio Weight Without Short (%)": "Weight (%)", "Fund": "ETF"},
        title="Tangency Portfolio Weight Distribution Without Short"
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

    # ---- GMVP Weights (centered) ----
    st.markdown("### Global Minimum Variance Portfolio (GMVP) Weights")

    col4, col5, col6 = st.columns([1, 2, 1])
    with col5:
        gmvp_df = pd.DataFrame({
            "Fund": mean_returns.index,
            "GMVP Weight (With short)": gmvp_weights,
            "GMVP Weight (Without short)": gmvp_weights_noshort
        })
        st.dataframe(
            gmvp_df.set_index("Fund").style.format("{:.4%}"),
            use_container_width=True
        )
    # ---- GMVP Weights Chart (Advanced) ----
    # Convert and sort GMVP weights
    gmvp_df["GMVP Weight With Short (%)"] = gmvp_df["GMVP Weight (With short)"] * 100
    gmvp_df_sorted = gmvp_df.sort_values(by="GMVP Weight With Short (%)", ascending=True)

    #  Plot GMVP weight distribution (horizontal bar chart)
    fig_weights = px.bar(
        gmvp_df_sorted,
        x="GMVP Weight With Short (%)",
        y="Fund",
        orientation="h",
        text="GMVP Weight With Short (%)",
        color="GMVP Weight With Short (%)",
        color_continuous_scale="Blues",
        labels={"GMVP Weight With Short (%)": "Weight (%)", "Fund": "ETF"},
        title="Global Minimum Variance Portfolio (GMVP) Weight Distribution With Short"
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
    
    
    # Convert and sort GMVP weights
    gmvp_df["GMVP Weight Without Short (%)"] = gmvp_df["GMVP Weight (Without short)"] * 100
    gmvp_df_sorted = gmvp_df.sort_values(by="GMVP Weight Without Short (%)", ascending=True)

    #  Plot GMVP weight distribution (horizontal bar chart)
    fig_weights = px.bar(
        gmvp_df_sorted,
        x="GMVP Weight Without Short (%)",
        y="Fund",
        orientation="h",
        text="GMVP Weight Without Short (%)",
        color="GMVP Weight Without Short (%)",
        color_continuous_scale="Blues",
        labels={"GMVP Weight Without Short (%)": "Weight (%)", "Fund": "ETF"},
        title="Global Minimum Variance Portfolio (GMVP) Weight Distribution Without Short"
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
