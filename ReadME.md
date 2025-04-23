
# Robo-Adviser: Bond ETF Portfolio Optimization Platform

This Streamlit-based web application is developed for academic purposes as part of the BMD5302 Financial Modeling course. The tool enables users to explore investment portfolio theory, visualize efficient frontiers, and generate optimal bond ETF allocations based on personalized risk preferences.

The project consists of two main modules:

- **Part 01:** Efficient Frontier Simulation and Visualization
- **Part 02:** Utility-Based Optimal Portfolio Based on Risk Aversion


---
## Deployment

This application is deployed and accessible via Streamlit Cloud:


🔗 **[Launch the Robo-Adviser App](https://robo-adviser-ee.streamlit.app/Risk_Aversion_Optimizer)**

---
---
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
- Application of the utility function:

\[
U(w) = w^T \mu - rac{A}{2} w^T \Sigma w
\]

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

## License

This project is intended for academic use only. Redistribution or commercial use is not permitted without explicit permission.


---

