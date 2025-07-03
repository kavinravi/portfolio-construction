import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from itertools import combinations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf

# Configure page
st.set_page_config(page_title="Portfolio Optimization Tool", layout="wide")

# Title and description
st.title("📊 Dynamic Factor-Based Portfolio Optimization")
st.markdown("This tool constructs and optimizes portfolios from user-defined tickers based on factor exposure targeting.")

# --- DATA LOADING & PROCESSING ---

@st.cache_data
def load_local_data(factor_map):
    """Load and process all ETF data from the data/ directory"""
    data_dir = 'data'
    try:
        data_files = [f for f in os.listdir(data_dir) if f.endswith('_advanced.csv')]
        if not data_files:
            st.error(f"No CSV files found in the '{data_dir}/'.")
            return None, None, None, None, None
    except FileNotFoundError:
        st.error(f"Data directory '{data_dir}/' not found.")
        return None, None, None, None, None

    tickers_map = {os.path.basename(f).split('_')[0]: os.path.join(data_dir, f) for f in data_files}
    
    # Check for factor ETFs
    for factor_tck in factor_map.values():
        if factor_tck not in tickers_map:
            st.error(f"Factor ETF '{factor_tck}' not found in local data files.")
            return None, None, None, None, None

    # Read all returns
    rets = {}
    for t, path in tickers_map.items():
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        rets[t] = df["Close"].pct_change().dropna()
    
    R_all = pd.DataFrame(rets).dropna()
    date_range = (R_all.index.min(), R_all.index.max())

    # Separate asset returns from factor returns
    asset_tickers = sorted(list(tickers_map.keys()))
    factor_tickers = sorted(list(set(factor_map.values())))

    R = R_all[asset_tickers]
    F_returns = R_all[factor_tickers]

    return R, F_returns, date_range, asset_tickers, factor_tickers

@st.cache_data
def load_dynamic_data(user_tickers, factor_tickers):
    """Load and process data for user-defined and factor tickers using yfinance"""
    
    all_tickers = sorted(list(set(user_tickers + factor_tickers)))
    if not all_tickers:
        return None, None, None, None, None

    try:
        # Download data
        data = yf.download(all_tickers, period="max", auto_adjust=True)['Close']
        if data.empty:
            st.error("Could not download any data. Please check ticker symbols.")
            return None, None, None, None, None

    except Exception as e:
        st.error(f"An error occurred during data download: {e}")
        return None, None, None, None, None

    # Calculate returns and drop missing values
    returns = data.pct_change().dropna()
    
    # Check for missing data
    if returns.empty:
        st.error("No overlapping data found for the selected tickers. Please choose a different set of tickers.")
        return None, None, None, None, None

    # Identify the actual date range
    date_range = (returns.index.min(), returns.index.max())

    # Separate user returns from factor returns
    R = returns[user_tickers]
    F_returns = returns[factor_tickers]
    
    return R, F_returns, date_range, user_tickers, factor_tickers

@st.cache_data
def calculate_exposures(R, F_returns, user_tickers, factor_names):
    """Calculate factor exposures for all ETFs"""
    exposures = pd.DataFrame(index=user_tickers, columns=factor_names, dtype=float)
    
    # Align dataframes to common dates
    aligned_data = pd.concat([R, F_returns], axis=1).dropna()
    R_aligned = aligned_data[user_tickers]
    F_aligned = aligned_data[F_returns.columns]

    for etf in exposures.index:
        y = R_aligned[etf].values.reshape(-1, 1)
        X = F_aligned.values
        lr = LinearRegression().fit(X, y)
        exposures.loc[etf] = lr.coef_.flatten()
    
    return exposures, R_aligned, F_aligned

# Optimization functions
def optimize_weights(E, target):
    """Optimize portfolio weights using least squares with constraints"""
    n = E.shape[0]
    
    # Weighted objective - penalize unwanted exposures more heavily
    def objective(w):
        actual = E.T.dot(w)
        errors = actual - target
        # 3x penalty for factors that should be zero
        weights = np.where(target == 0, 3.0, 1.0)
        return np.sum(weights * errors**2)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
    ]
    
    # Add hard constraints for zero-target factors  
    for i in range(len(target)):
        if target[i] == 0:
            # Limit unwanted factor exposure to max 5%
            def make_constraint(idx):
                return lambda w: 0.05 - abs(E[:, idx].T.dot(w))
            constraints.append({
                'type': 'ineq', 
                'fun': make_constraint(i)
            })
    
    bounds = [(0, 1)] * n
    x0 = np.ones(n) / n
    
    try:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not res.success:
            # Fall back to original method if constrained optimization fails
            def simple_objective(w):
                return np.sum((E.T.dot(w) - target)**2)
            simple_cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            res = minimize(simple_objective, x0, method="SLSQP", bounds=bounds, constraints=simple_cons)
            if not res.success:
                # If still failing, raise the error to be caught
                raise RuntimeError("Optimization failed: " + res.message)
        return res.x
    except Exception as e:
        # st.warning(f"Optimization failed for a subset. Details: {e}")
        return None # Return None on failure

def score_portfolio(etf_subset, exposures, target):
    """Score a portfolio subset"""
    E_sub = exposures.loc[list(etf_subset)].values
    w = optimize_weights(E_sub, target)
    
    # If optimization fails, return infinite error
    if w is None:
        return None, None, np.inf
        
    port_exp = E_sub.T.dot(w)
    error = np.linalg.norm(port_exp - target)
    return w, port_exp, error

def find_best_subset(exposures, target, n):
    """Find the best n-ETF subset using a greedy forward selection algorithm."""
    
    # Start with an empty set
    best_set = []
    best_error = np.inf

    # Get the full list of available ETFs
    all_etfs = list(exposures.index)
    
    # Status indicators
    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(n):
        status_text.text(f"Selecting ETF {i + 1} of {n}...")
        
        # Find the best ETF to add to the current set
        best_etf_to_add = None
        current_best_error_for_step = np.inf
        
        # Iterate through the remaining ETFs
        remaining_etfs = [etf for etf in all_etfs if etf not in best_set]
        
        for etf_to_add in remaining_etfs:
            # Create a temporary set to evaluate
            temp_set = best_set + [etf_to_add]
            
            # Score this temporary portfolio
            _, _, error = score_portfolio(temp_set, exposures, target)
            
            if error < current_best_error_for_step:
                current_best_error_for_step = error
                best_etf_to_add = etf_to_add
        
        # Add the best ETF found in this step to our portfolio
        if best_etf_to_add:
            best_set.append(best_etf_to_add)
            best_error = current_best_error_for_step
        
        # Update progress
        progress_bar.progress((i + 1) / n)

    # Final scoring of the selected portfolio
    w, port_exp, error = score_portfolio(best_set, exposures, target)
    
    status_text.empty()
    progress_bar.empty()
    
    return best_set, w, error

# Performance metrics calculation
def calculate_performance_metrics(returns, benchmark_returns):
    """Calculate comprehensive performance metrics"""
    # Convert to pandas series if numpy arrays
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)
    
    # Basic metrics
    total_return_port = (1 + returns).cumprod().iloc[-1] - 1
    total_return_bench = (1 + benchmark_returns).cumprod().iloc[-1] - 1
    
    # Annualized metrics (assuming daily data)
    trading_days = 252
    years = len(returns) / trading_days
    
    ann_return_port = (1 + total_return_port) ** (1/years) - 1
    ann_return_bench = (1 + total_return_bench) ** (1/years) - 1
    
    ann_vol_port = returns.std() * np.sqrt(trading_days)
    ann_vol_bench = benchmark_returns.std() * np.sqrt(trading_days)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_port = ann_return_port / ann_vol_port if ann_vol_port > 0 else 0
    sharpe_bench = ann_return_bench / ann_vol_bench if ann_vol_bench > 0 else 0
    
    # Beta calculation
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    variance_bench = np.var(benchmark_returns)
    beta = covariance / variance_bench if variance_bench > 0 else 0
    
    # Maximum drawdown
    cum_port = (1 + returns).cumprod()
    rolling_max_port = cum_port.expanding().max()
    drawdown_port = (cum_port - rolling_max_port) / rolling_max_port
    max_drawdown_port = drawdown_port.min()
    
    cum_bench = (1 + benchmark_returns).cumprod()
    rolling_max_bench = cum_bench.expanding().max()
    drawdown_bench = (cum_bench - rolling_max_bench) / rolling_max_bench
    max_drawdown_bench = drawdown_bench.min()
    
    # Information ratio
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(trading_days)
    information_ratio = (ann_return_port - ann_return_bench) / tracking_error if tracking_error > 0 else 0
    
    return {
        'portfolio': {
            'total_return': total_return_port,
            'annual_return': ann_return_port,
            'volatility': ann_vol_port,
            'sharpe_ratio': sharpe_port,
            'max_drawdown': max_drawdown_port,
            'beta': beta
        },
        'benchmark': {
            'total_return': total_return_bench,
            'annual_return': ann_return_bench,
            'volatility': ann_vol_bench,
            'sharpe_ratio': sharpe_bench,
            'max_drawdown': max_drawdown_bench
        },
        'relative': {
            'excess_return': ann_return_port - ann_return_bench,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    }

# Main application
def main():
    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Portfolio & Factor Configuration")

    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ('Local Directory', 'Dynamic Ticker Search'),
        index=1
    )

    # Factor definitions (proxies)
    factor_map = {
        "us": "SPY", "intl": "ACWI", "small": "IJR", "large": "IVV",
        "value": "IVE", "growth": "IVW", "momentum": "SPMO", "quality": "QUAL",
    }
    factor_tickers = sorted(list(set(factor_map.values())))
    factor_names = sorted(list(factor_map.keys()))

    # Asset selection UI
    st.sidebar.subheader("Asset Universe")
    if data_source == 'Dynamic Ticker Search':
        default_tickers = "SPY, QQQ, IWM, GLD, TLT, HYG, EEM, VNQ"
        user_ticker_input = st.sidebar.text_area("Enter tickers (comma-separated)", value=default_tickers, height=100)
        asset_tickers = sorted([t.strip().upper() for t in user_ticker_input.split(',') if t.strip()])
    else: # Local Directory
        # In local mode, asset tickers are discovered by the loader
        st.sidebar.info("Assets are loaded from the `data/` directory.")
        asset_tickers = [] # Placeholder, will be populated by loader

    # Factor weights input
    st.sidebar.subheader("Target Factor Weights")
    factor_weights = {}
    defaults = {"us": 0.5, "large": 0.3, "quality": 0.2} # Example defaults
    
    for factor in factor_names:
        factor_weights[factor] = st.sidebar.number_input(
            f"{factor.title()} Weight", 
            min_value=0.0, max_value=1.0, 
            value=defaults.get(factor, 0.0), 
            step=0.05, key=f"weight_{factor}"
        )
    
    total_weight = sum(factor_weights.values())
    if abs(total_weight - 1.0) > 0.001:
        st.sidebar.error(f"⚠️ Total weight: {total_weight:.3f} (must sum to 1.0)")
    else:
        st.sidebar.success(f"✅ Total weight: {total_weight:.3f}")

    # Number of assets in portfolio
    st.sidebar.subheader("Portfolio Size")
    n_etfs = st.sidebar.number_input(
        "Number of Assets in Portfolio", 
        min_value=1, max_value=20, 
        value=3, 
        step=1,
        help="Maximum will be adjusted based on available assets"
    )

    # --- MAIN LOGIC ---
    if st.sidebar.button("🚀 Optimize Portfolio", type="primary"):
        if data_source == 'Dynamic Ticker Search' and not asset_tickers:
            st.warning("Please enter at least one ticker.")
            return
        if abs(total_weight - 1.0) > 0.001:
            st.error("Please ensure factor weights sum to 1.0 before optimizing.")
            return

        # Load data based on selected source
        with st.spinner("Loading and processing data..."):
            if data_source == 'Dynamic Ticker Search':
                R, F_returns, date_range, asset_tickers, _ = load_dynamic_data(asset_tickers, factor_tickers)
            else: # Local Directory
                R, F_returns, date_range, asset_tickers, _ = load_local_data(factor_map)

        if R is None:
            return # Stop if data loading failed

        st.success(f"Data loaded for {len(asset_tickers)} assets from {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")

        # Validate n_etfs against available assets
        if n_etfs > len(asset_tickers):
            st.warning(f"Requested {n_etfs} assets but only {len(asset_tickers)} are available. Using {len(asset_tickers)} assets.")
            n_etfs = len(asset_tickers)

        # Calculate exposures
        with st.spinner("Calculating factor exposures..."):
            factor_returns_aligned_to_map = pd.DataFrame({name: F_returns[ticker] for name, ticker in factor_map.items()}).dropna()
            exposures, R_aligned, F_aligned = calculate_exposures(R, factor_returns_aligned_to_map, asset_tickers, factor_names)

        # Run optimization
        target = np.array([factor_weights[factor] for factor in factor_names])
        with st.spinner(f"Finding optimal {n_etfs}-asset portfolio..."):
            combo, weights, error = find_best_subset(exposures, target, n_etfs)
        
        # Store results
        st.session_state.optimization_results = {
            'combo': combo, 'weights': weights, 'error': error, 'target': target,
            'factor_names': factor_names, 'R_aligned': R_aligned, 'F_aligned': F_aligned,
            'exposures': exposures
        }
        st.success("Optimization completed!")

    # --- DISPLAY RESULTS ---
    if 'optimization_results' in st.session_state:
        results = st.session_state.optimization_results
        combo, weights, error, target, factor_names = results['combo'], results['weights'], results['error'], results['target'], results['factor_names']
        R_aligned, F_aligned, exposures = results['R_aligned'], results['F_aligned'], results['exposures']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio Results", "📊 Performance Analysis", "🔬 Advanced Analytics", "🧠 Prediction"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimized Portfolio")
                portfolio_df = pd.DataFrame({'Asset': combo, 'Weight': weights, 'Weight (%)': weights * 100})
                st.dataframe(portfolio_df, hide_index=True)
                st.metric("Optimization Error", f"{error:.6f}")
                
                st.subheader("Factor Exposure vs Target")
                actual_exposure = exposures.loc[list(combo)].T.dot(weights)
                comparison_df = pd.DataFrame({
                    'Factor': factor_names,
                    'Target': target,
                    'Actual': actual_exposure,
                    'Difference': actual_exposure - target
                })
                st.dataframe(comparison_df, hide_index=True)
            
            with col2:
                port_rets = R_aligned[list(combo)].dot(weights)
                spy_rets = R_aligned["SPY"] if "SPY" in R_aligned.columns else F_aligned['us']
                metrics = calculate_performance_metrics(port_rets, spy_rets)
                
                st.subheader("Performance Summary")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Portfolio Annual Return", f"{metrics['portfolio']['annual_return']:.2%}")
                    st.metric("Portfolio Volatility", f"{metrics['portfolio']['volatility']:.2%}")
                    st.metric("Portfolio Sharpe Ratio", f"{metrics['portfolio']['sharpe_ratio']:.3f}")
                    st.metric("Portfolio Beta", f"{metrics['portfolio']['beta']:.3f}")
                with col_b:
                    st.metric("SPY Annual Return", f"{metrics['benchmark']['annual_return']:.2%}")
                    st.metric("SPY Volatility", f"{metrics['benchmark']['volatility']:.2%}")
                    st.metric("SPY Sharpe Ratio", f"{metrics['benchmark']['sharpe_ratio']:.3f}")
                    st.metric("Excess Return", f"{metrics['relative']['excess_return']:.2%}")
        
        with tab2:
            st.subheader("Cumulative Performance")
            port_rets = R_aligned[list(combo)].dot(weights)
            spy_rets = R_aligned["SPY"] if "SPY" in R_aligned.columns else F_aligned['us']
            initial_capital = 1000000
            cum_portfolio = (1 + port_rets).cumprod() * initial_capital
            cum_spy = (1 + spy_rets).cumprod() * initial_capital
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(cum_portfolio.index, cum_portfolio.values, label=f"Optimized {len(combo)}-Asset Portfolio", linewidth=2)
            ax.plot(cum_spy.index, cum_spy.values, label="SPY (Benchmark)", linewidth=2)
            ax.set_title("Cumulative Portfolio Value", fontsize=14, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Detailed Performance Metrics")
            metrics = calculate_performance_metrics(port_rets, spy_rets)
            
            # Create a clean, readable DataFrame for metrics
            p_metrics = metrics['portfolio']
            b_metrics = metrics['benchmark']
            
            metrics_data = {
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Volatility',
                    'Sharpe Ratio',
                    'Maximum Drawdown',
                    'Beta (vs SPY)'
                ],
                'Portfolio': [
                    f"{p_metrics['total_return']:.2%}",
                    f"{p_metrics['annual_return']:.2%}",
                    f"{p_metrics['volatility']:.2%}",
                    f"{p_metrics['sharpe_ratio']:.3f}",
                    f"{p_metrics['max_drawdown']:.2%}",
                    f"{p_metrics['beta']:.3f}"
                ],
                'SPY Benchmark': [
                    f"{b_metrics['total_return']:.2%}",
                    f"{b_metrics['annual_return']:.2%}",
                    f"{b_metrics['volatility']:.2%}",
                    f"{b_metrics['sharpe_ratio']:.3f}",
                    f"{b_metrics['max_drawdown']:.2%}",
                    "1.000"
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, hide_index=True)

        with tab3:
            st.subheader("Factor Exposure Matrix")
            st.dataframe(exposures.round(4))
            
            st.subheader("Correlation Matrix")
            assets_for_corr = list(combo)
            if 'SPY' not in assets_for_corr and 'SPY' in R_aligned.columns:
                assets_for_corr.append('SPY')
            correlation_matrix = R_aligned[assets_for_corr].corr()
            st.dataframe(correlation_matrix.round(3))
            
            st.subheader("Rolling Returns (30-day)")
            port_rets = R_aligned[list(combo)].dot(weights)
            spy_rets = R_aligned["SPY"] if "SPY" in R_aligned.columns else F_aligned['us']
            rolling_returns = pd.DataFrame({
                'Portfolio': port_rets.rolling(30).mean() * 30,
                'SPY': spy_rets.rolling(30).mean() * 30
            }).dropna()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(rolling_returns.index, rolling_returns['Portfolio'], label='Portfolio', alpha=0.8)
            ax.plot(rolling_returns.index, rolling_returns['SPY'], label='SPY', alpha=0.8)
            ax.set_title("30-Day Rolling Returns")
            ax.set_ylabel("Return")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with tab4:
            st.subheader("Experimental Monthly Return Prediction")
            st.warning("Disclaimer: This prediction is for informational and academic purposes only. It is based on a simplified model and past performance, which is not indicative of future results. This is not financial advice.")

            port_rets = R_aligned[list(combo)].dot(weights)
            spy_rets = R_aligned["SPY"] if "SPY" in R_aligned.columns else F_aligned['us']

            with st.spinner("Training prediction model... This may take a moment."):
                look_back = 30 # Use last 30 days to predict the next day
                test_preds, future_preds = train_and_predict(port_rets, look_back=look_back)
            
            st.success("Model training complete.")

            # --- VISUALIZATION ---
            
            # 1. Predicted vs. Actual Plot - Cumulative Returns
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            # We need to align the test predictions with the actual returns
            actual_returns_for_plot = port_rets.values[look_back + 1:]
            dates_for_plot = port_rets.index[look_back + 1:]
            spy_returns_for_plot = spy_rets.values[look_back + 1:]

            # Convert to cumulative returns for cleaner visualization
            actual_cumulative = (1 + pd.Series(actual_returns_for_plot)).cumprod()
            predicted_cumulative = (1 + pd.Series(test_preds)).cumprod()
            spy_cumulative = (1 + pd.Series(spy_returns_for_plot)).cumprod()

            ax1.plot(dates_for_plot, actual_cumulative, label="Actual Portfolio Cumulative Returns", color='blue', linewidth=2)
            ax1.plot(dates_for_plot, predicted_cumulative, label="Predicted Cumulative Returns", color='orange', linestyle='--', linewidth=2)
            ax1.plot(dates_for_plot, spy_cumulative, label="SPY Cumulative Returns", color='green', linewidth=2)
            ax1.set_title("Model Fit: Actual vs. Predicted Cumulative Returns", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Cumulative Return (Starting from 1)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)

            # 2. Future Prediction Plot
            future_dates = pd.to_datetime(port_rets.index[-1]) + pd.to_timedelta(np.arange(1, len(future_preds) + 1), 'd')
            
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(future_dates, np.cumsum(future_preds), label="Predicted Cumulative Return", marker='o')
            ax2.set_title("Forecast: Next Month Cumulative Return", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Cumulative Return")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

            # 3. Summary Metrics
            predicted_monthly_return = np.sum(future_preds)
            st.metric("Predicted Cumulative Return for Next Month", f"{predicted_monthly_return:.2%}")
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimized Portfolio")
                portfolio_df = pd.DataFrame({'Asset': combo, 'Weight': weights, 'Weight (%)': weights * 100})
                st.dataframe(portfolio_df, hide_index=True)
                st.metric("Optimization Error", f"{error:.6f}")
                
                st.subheader("Factor Exposure vs Target")
                actual_exposure = exposures.loc[list(combo)].T.dot(weights)
                comparison_df = pd.DataFrame({
                    'Factor': factor_names,
                    'Target': target,
                    'Actual': actual_exposure,
                    'Difference': actual_exposure - target
                })
                st.dataframe(comparison_df, hide_index=True)
            
            with col2:
                port_rets = R_aligned[list(combo)].dot(weights)
                spy_rets = R_aligned["SPY"] if "SPY" in R_aligned.columns else F_aligned['us']
                metrics = calculate_performance_metrics(port_rets, spy_rets)
                
                st.subheader("Performance Summary")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Portfolio Annual Return", f"{metrics['portfolio']['annual_return']:.2%}")
                    st.metric("Portfolio Volatility", f"{metrics['portfolio']['volatility']:.2%}")
                    st.metric("Portfolio Sharpe Ratio", f"{metrics['portfolio']['sharpe_ratio']:.3f}")
                    st.metric("Portfolio Beta", f"{metrics['portfolio']['beta']:.3f}")
                with col_b:
                    st.metric("SPY Annual Return", f"{metrics['benchmark']['annual_return']:.2%}")
                    st.metric("SPY Volatility", f"{metrics['benchmark']['volatility']:.2%}")
                    st.metric("SPY Sharpe Ratio", f"{metrics['benchmark']['sharpe_ratio']:.3f}")
                    st.metric("Excess Return", f"{metrics['relative']['excess_return']:.2%}")
        
        with tab2:
            st.subheader("Cumulative Performance")
            port_rets = R_aligned[list(combo)].dot(weights)
            spy_rets = R_aligned["SPY"] if "SPY" in R_aligned.columns else F_aligned['us']
            initial_capital = 1000000
            cum_portfolio = (1 + port_rets).cumprod() * initial_capital
            cum_spy = (1 + spy_rets).cumprod() * initial_capital
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(cum_portfolio.index, cum_portfolio.values, label=f"Optimized {len(combo)}-Asset Portfolio", linewidth=2)
            ax.plot(cum_spy.index, cum_spy.values, label="SPY (Benchmark)", linewidth=2)
            ax.set_title("Cumulative Portfolio Value", fontsize=14, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Detailed Performance Metrics")
            metrics = calculate_performance_metrics(port_rets, spy_rets)
            
            # Create a clean, readable DataFrame for metrics
            p_metrics = metrics['portfolio']
            b_metrics = metrics['benchmark']
            
            metrics_data = {
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Volatility',
                    'Sharpe Ratio',
                    'Maximum Drawdown',
                    'Beta (vs SPY)'
                ],
                'Portfolio': [
                    f"{p_metrics['total_return']:.2%}",
                    f"{p_metrics['annual_return']:.2%}",
                    f"{p_metrics['volatility']:.2%}",
                    f"{p_metrics['sharpe_ratio']:.3f}",
                    f"{p_metrics['max_drawdown']:.2%}",
                    f"{p_metrics['beta']:.3f}"
                ],
                'SPY Benchmark': [
                    f"{b_metrics['total_return']:.2%}",
                    f"{b_metrics['annual_return']:.2%}",
                    f"{b_metrics['volatility']:.2%}",
                    f"{b_metrics['sharpe_ratio']:.3f}",
                    f"{b_metrics['max_drawdown']:.2%}",
                    "1.000"
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, hide_index=True)

        with tab3:
            st.subheader("Factor Exposure Matrix")
            st.dataframe(exposures.round(4))
            
            st.subheader("Correlation Matrix")
            assets_for_corr = list(combo)
            if 'SPY' not in assets_for_corr and 'SPY' in R_aligned.columns:
                assets_for_corr.append('SPY')
            correlation_matrix = R_aligned[assets_for_corr].corr()
            st.dataframe(correlation_matrix.round(3))
            
            st.subheader("Rolling Returns (30-day)")
            port_rets = R_aligned[list(combo)].dot(weights)
            spy_rets = R_aligned["SPY"] if "SPY" in R_aligned.columns else F_aligned['us']
            rolling_returns = pd.DataFrame({
                'Portfolio': port_rets.rolling(30).mean() * 30,
                'SPY': spy_rets.rolling(30).mean() * 30
            }).dropna()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(rolling_returns.index, rolling_returns['Portfolio'], label='Portfolio', alpha=0.8)
            ax.plot(rolling_returns.index, rolling_returns['SPY'], label='SPY', alpha=0.8)
            ax.set_title("30-Day Rolling Returns")
            ax.set_ylabel("Return")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

# --- PREDICTION MODEL ---

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for time-series prediction."""
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def create_dataset(data, look_back=1):
    """Create dataset for time-series forecasting."""
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

@st.cache_data
def train_and_predict(portfolio_returns, look_back=30, epochs=100, lr=0.001):
    """Train an MLP model and predict future returns."""
    # Prepare data
    returns_np = portfolio_returns.values.astype('float32').reshape(-1, 1)
    train_X, train_Y = create_dataset(returns_np, look_back)
    
    # Convert to PyTorch tensors
    train_X = torch.from_numpy(train_X)
    train_Y = torch.from_numpy(train_Y).view(-1, 1)

    # Define and train the model
    model = MLP(input_size=look_back, hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_Y)
        loss.backward()
        optimizer.step()

    # Generate predictions on the training data for comparison
    model.eval()
    with torch.no_grad():
        test_predictions = model(train_X).numpy().flatten()

    # Predict the next month (21 trading days)
    future_predictions = []
    last_sequence = torch.from_numpy(returns_np[-look_back:].flatten().astype('float32'))

    with torch.no_grad():
        for _ in range(21):
            pred = model(last_sequence.view(1, -1))
            future_predictions.append(pred.item())
            # Update the sequence with the new prediction
            last_sequence = torch.roll(last_sequence, -1)
            last_sequence[-1] = pred
            
    return test_predictions, future_predictions

if __name__ == "__main__":
    main() 