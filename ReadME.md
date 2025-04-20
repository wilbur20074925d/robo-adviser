# 📊 ETF Robo-Adviser Platform

This is a two-part Streamlit web application designed for academic portfolio analysis and optimization.

## 🚀 Features

### 🧩 Part 1: Efficient Frontier Analyzer
- Upload ETF price data
- Calculate annualized returns, volatility, Sharpe Ratio
- Visualize efficient frontier (with and without short sales)
- Plot Global Minimum Variance Portfolio (GMVP)
- Display CML and tangency portfolio

### 🎯 Part 2: Risk Aversion Optimizer
- Answer a short questionnaire to determine investor risk aversion (A)
- Compute optimal portfolio via utility maximization:
  
  \[
  U = r - \frac{A \cdot \sigma^2}{2}
  \]

- Display optimal weights, expected return, volatility, utility
- Pie chart visualization of non-negative weights
- Session history tracking for multiple runs

## 📁 Project Structure