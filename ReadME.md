
# BMD5302 Group Project Robo-Adviser: ETF Portfolio Optimization Platform

This Streamlit-based web application is developed for academic purposes as part of the BMD5302 Financial Modeling course. The tool enables users to explore investment portfolio theory, visualize efficient frontiers, and generate optimal bond ETF allocations based on personalized risk preferences.

The project consists of two main modules:

- **Part 01:** Efficient Frontier Simulation and Visualization
- **Part 02:** Utility-Based Optimal Portfolio Based on Risk Aversion


---
## Deployment

This application is deployed and accessible via Streamlit Cloud:


🔗 **[Launch the Robo-Adviser App](https://robo-adviser-ee.streamlit.app/Risk_Aversion_Optimizer)**

---
## Team Members
- WANG Xinyu
- YANG Yuebo
- ZENG Rui
- ZHOU Lingxiang

## Features

- Upload real-world bond ETF price data
- Simulate and visualize the Efficient Frontier (with and without short sales)
- Calculate Global Minimum Variance Portfolio (GMVP)
- Estimate investor risk aversion via an interactive questionnaire
- Generate personalized optimal portfolios using a utility function
- View allocation breakdowns via pie charts and interactive metrics
- Track session history and support multiple use scenarios

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/robo-adviser.git
cd robo-adviser
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Run the Web Application

```bash
streamlit run Home.py
```

---

## Folder Structure

```
robo-adviser/
├── Home.py                         # Main navigation page
├── pages/
│   ├── 1_Efficient_Frontier.py     # Efficient Frontier Simulator
│   └── 2_Risk_Aversion_Optimizer.py# Risk-Based Optimizer Module
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
```

---

## 📘 Module Details

### Part 01: Efficient Frontier

This module allows the user to:

- Upload a CSV file of **daily adjusted closing prices** for 10 bond ETFs.
- Compute:
  - Annualized return
  - Annualized volatility
  - Sharpe ratio
  - Covariance matrix
- Simulate 5,000 random portfolios (with and without short selling).
- Plot:
  - Random portfolios
  - Efficient Frontier
  - Global Minimum Variance Portfolio (GMVP)
  - Tangency Portfolio (maximum Sharpe)
- View:
  - Interactive charts
  - Heatmaps of correlations
  - Top 10 frontier points
  - Return distributions and metrics

#### Input Format (CSV):

```text
Date,AGG,BND,TLT,IEF,SHY,LQD,HYG,TIP,MUB,EMB
2015-01-01,101.23,103.45,...
2015-01-02,101.40,103.55,...
...
```

---

### Part 02: Risk Aversion Optimizer

This module helps investors find an optimal portfolio that maximizes their utility given a subjective risk aversion score. It features:

- A behavioral questionnaire (6 questions)
- Calculation of a **risk aversion coefficient (A)** based on responses
- Portfolio optimization (with or without short sales)
- Output of:
  - Expected return
  - Portfolio volatility
  - Utility score
  - Weight distribution (table + pie chart)

- Session tracking for multiple runs

---

## Sample Data

You can use any historical bond ETF dataset or download from Yahoo Finance. Example tickers:

- AGG, BND, TLT, IEF, SHY, LQD, HYG, TIP, MUB, EMB

---

## Notes

- This platform is built for educational and demonstration purposes.
- The optimization methods used are simplified and may not reflect real-world constraints.
- Consider extending the platform with backtesting, ESG filtering, and rebalancing simulation.


---

## Calculation Methodology

This section outlines the theoretical foundations and mathematical formulas employed in the two components of the Robo-Adviser platform: the Efficient Frontier construction based on Modern Portfolio Theory (Part 01), and the utility-maximizing optimal portfolio generation based on investor risk preferences (Part 02).

---

### Part 01: Efficient Frontier – Modern Portfolio Theory (MPT)

The efficient frontier represents the set of optimal portfolios offering the highest expected return for a given level of risk. The analysis assumes asset returns are normally distributed and that investors seek to maximize return while minimizing risk, measured by variance.

Given a portfolio of $n$ assets:

- Let $\mu \in \mathbb{R}^n$ be the vector of expected annualized returns
- Let $\Sigma \in \mathbb{R}^{n \times n}$ be the covariance matrix of asset returns
- Let $w \in \mathbb{R}^n$ be the portfolio weights such that $\sum w_i = 1$

The core calculations are as follows:

#### Portfolio Expected Return


$$\mu_p = w^\top \mu$$


#### Portfolio Risk (Volatility)


$$\sigma_p = \sqrt{w^\top \Sigma w}$$


#### Sharpe Ratio

Assuming a constant risk-free rate $ r_f $, the Sharpe ratio is:


$$\text{Sharpe Ratio} = \frac{\mu_p - r_f}{\sigma_p}$$


#### Global Minimum Variance Portfolio (GMVP)

The GMVP minimizes portfolio variance irrespective of return preferences. Its closed-form solution is:


$$w_{GMVP} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}$$


Where $\mathbf{1}$ is a vector of ones.

#### Efficient Frontier Construction

To construct the frontier, we simulate thousands of random portfolios under:

- **Short-sales allowed**: $w_i \in \mathbb{R}$, $sum w_i = 1$
- **Short-sales disallowed**: $w_i \geq 0$, $\sum w_i = 1$

The frontier is the upper envelope of portfolios with the highest return for a given level of risk. We also identify the **Tangency Portfolio** as the point with the maximum Sharpe ratio.

---

### Part 02: Utility-Based Optimal Portfolio – Risk Aversion Optimization

This section adopts a **mean-variance utility function** to compute the investor's optimal portfolio, conditioned on a subjective **risk aversion score $A$**. This framework assumes that the investor's utility increases with expected return and decreases with variance.

#### Utility Function


$$U(w) = w^\top \mu - \frac{A}{2} w^\top \Sigma w$$


Where:

- $\mu$: vector of expected returns
- $\Sigma$: covariance matrix of returns
- $A$: investor-specific risk aversion coefficient

#### Optimal Weights

If short selling is allowed, the optimal weights are proportional to the excess expected return scaled by risk:


$$w^* = \frac{\Sigma^{-1} \mu}{\sum |\Sigma^{-1} \mu|}$$


If short selling is disallowed, all negative values in $w^*$ are clipped to zero, and the remaining weights are normalized:


$$w^* = \frac{\max(0, \Sigma^{-1} \mu)}{\sum \max(0, \Sigma^{-1} \mu)}$$


#### Portfolio Metrics under Optimal Weights

- **Expected Return**:


$$\mu_p = w^{*\top} \mu$$


- **Volatility**:


$$\sigma_p = \sqrt{w^{*\top} \Sigma w^*}$$

- **Utility Score**:


$$U = \mu_p - \frac{A}{2} \sigma_p^2$$


#### Risk Aversion Coefficient $A$

The value of $ A $ is determined via a six-question behavioral questionnaire. Higher values of $A$ correspond to more risk-averse profiles. The platform allows users to experiment with different values of$ A$ to observe how optimal allocations shift with investor preferences.


---
## License

This project is intended for academic use only. Redistribution or commercial use is not permitted without explicit permission.


---

