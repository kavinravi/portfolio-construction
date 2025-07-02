# Factor-Based Portfolio Optimization Tool

A simple, interactive Streamlit app that helps you build ETF portfolios by targeting specific factor exposure and then compares your optimized blend against the SPY benchmark.

---

## Overview

Traditional investors know that broad-market ETFs (like SPY) capture the overall return, but smart tilts into proven “factors” can improve risk-adjusted performance. This tool:

- **Loads** your historical ETF data (daily closes)
- **Estimates** each ETF’s exposure to common factors
- **Optimizes** a portfolio of N ETFs to match your chosen mix of factor weights
- **Visualizes** portfolio composition, factor exposures, and performance vs. SPY

No heavy programming required. Just clone, install the requirements, and launch the app. Or, for an even easier start, just visit the cloud deployment link (below). 

---

## Quick Start

1. **Visit the live app**\
   [https://portfolio-construction.streamlit.app/](https://portfolio-construction.streamlit.app/)

2. **...Or clone the repo (manual method)**

   ```bash
   git clone https://github.com/kavinravi/portfolio-construction.git
   cd portfolio-construction
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run portfolio.py
   ```

Use the sidebar to set factor weights and the number of ETFs, then click **Optimize Portfolio**.

---

## How It Works (High Level)

1. **Factor Exposures**

   - The app treats each ETF’s daily returns as a blend of standard factors: **US equity**, **International**, **Small-cap**, **Value**, **Growth**, **Momentum**, **Quality**.
   - It runs a linear regression to estimate each ETF’s factor loadings.

2. **Portfolio Optimization**

   - You set your target mix (e.g., 40% value, 20% growth, 20% momentum, 20% quality).
   - Choose how many ETFs (n) to include.
   - The app evaluates combinations and weightings to minimize the gap between actual vs. target exposures, under a 0–100% weight constraint summing to 100%.

3. **Performance Analysis**

   - Benchmarks your optimized portfolio against SPY on metrics like total return, annualized return, volatility, Sharpe ratio, drawdown, beta, and information ratio.
   - Plots the cumulative growth of a \$1 M starting capital over the backtest period.

---

## Configuration & Customization

- **Factor Weights**: Adjust sliders in the sidebar; they must sum to 1.
- **# of ETFs**: From 1 up to the full list of supported tickers.
- **Data Range**: Swap in your own CSVs (e.g., extend beyond 2025) as long as filenames follow the existing pattern.

**To add a new ETF**:

1. Place its CSV into `data/`.
2. Add its ticker and file path in the `tickers` dict inside `load_data()`.
3. (Optional) Assign it a role in the `factors` dict if it serves as a factor proxy.

---

## Limitations & Future Features

- **Backtest Horizon**: Data covers \~10 years and ends mid-June 2025, limiting historical context—results may vary outside this window.
- **Predictive Analytics**: Upcoming releases will introduce forecasting modules and predictive features to enhance forward-looking insights.

---



