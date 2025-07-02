import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from itertools import combinations
import os

# Configure page
st.set_page_config(page_title="Portfolio Optimization Tool", layout="wide")

# Title and description
st.title("📊 Factor-Based Portfolio Optimization")
st.markdown("This tool optimizes ETF portfolios based on factor exposure targeting using least squares optimization.")

# Load data function
@st.cache_data
def load_data():
    """Load and process all ETF data"""
    
    # Define tickers and their CSV filenames (updated paths)
    tickers = {
        "ACWI": "data/ACWI_2016-06-17_to_2025-06-16_advanced.csv",
        "IJR": "data/IJR_2016-06-17_to_2025-06-16_advanced.csv",
        "IVE": "data/IVE_2016-06-17_to_2025-06-16_advanced.csv",
        "IVV": "data/IVV_2016-06-17_to_2025-06-16_advanced.csv",
        "IVW": "data/IVW_2016-06-17_to_2025-06-16_advanced.csv",
        "QUAL": "data/QUAL_2016-06-17_to_2025-06-16_advanced.csv",
        "SPMO": "data/SPMO_2016-06-17_to_2025-06-16_advanced.csv",
        "SPY": "data/SPY_2016-06-17_to_2025-06-17_advanced.csv",
        "XLB": "data/XLB_2016-06-17_to_2025-06-17_advanced.csv",
        "XLC": "data/XLC_2016-06-17_to_2025-06-17_advanced.csv",
        "XLE": "data/XLE_2016-06-17_to_2025-06-17_advanced.csv",
        "XLF": "data/XLF_2016-06-17_to_2025-06-17_advanced.csv",
        "XLI": "data/XLI_2016-06-17_to_2025-06-17_advanced.csv",
        "XLK": "data/XLK_2016-06-17_to_2025-06-17_advanced.csv",
        "XLP": "data/XLP_2016-06-17_to_2025-06-17_advanced.csv",
        "XLRE": "data/XLRE_2016-06-17_to_2025-06-17_advanced.csv",
        "XLU": "data/XLU_2016-06-17_to_2025-06-17_advanced.csv",
        "XLV": "data/XLV_2016-06-17_to_2025-06-17_advanced.csv",
        "XLY": "data/XLY_2016-06-17_to_2025-06-17_advanced.csv"
    }
    
    # Factor proxies mapping
    factors = {
        "us": "SPY",
        "intl": "ACWI",
        "small": "IJR",
        "large": "SPY",
        "value": "IVE",
        "growth": "IVW",
        "momentum": "SPMO",
        "quality": "QUAL",
    }
    
    # Read and align all returns
    rets = {}
    for t, path in tickers.items():
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            rets[t] = df["Close"].pct_change().dropna()
        else:
            st.error(f"File not found: {path}")
            return None, None, None, None
    
    R = pd.DataFrame(rets).dropna()
    
    # Build factor-return DataFrame F
    F = pd.DataFrame({f: R[tck] for f, tck in factors.items()}).dropna()
    
    # Align R and F
    df = pd.concat([R, F], axis=1).dropna()
    R_aligned = df[list(rets.keys())]
    F_aligned = df[list(factors.keys())]
    
    return tickers, factors, R_aligned, F_aligned

# Calculate exposures
@st.cache_data
def calculate_exposures(R, F, tickers, factors):
    """Calculate factor exposures for all ETFs"""
    exposures = pd.DataFrame(index=tickers.keys(), columns=factors.keys(), dtype=float)
    
    for etf in exposures.index:
        y = R[etf].values.reshape(-1, 1)
        X = F.values
        lr = LinearRegression().fit(X, y)
        exposures.loc[etf] = lr.coef_.flatten()
    
    return exposures

# Optimization functions
def optimize_weights(E, target):
    """Optimize portfolio weights using least squares"""
    n = E.shape[0]
    def objective(w):
        return np.sum((E.T.dot(w) - target)**2)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    x0 = np.ones(n) / n
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)
    return res.x

def score_portfolio(etf_subset, exposures, target):
    """Score a portfolio subset"""
    E_sub = exposures.loc[list(etf_subset)].values
    w = optimize_weights(E_sub, target)
    port_exp = E_sub.T.dot(w)
    error = np.linalg.norm(port_exp - target)
    return w, port_exp, error

def find_best_subset(exposures, target, n):
    """Find the best n-ETF subset"""
    best = (None, None, np.inf)
    total_combinations = len(list(combinations(exposures.index, n)))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, combo in enumerate(combinations(exposures.index, n)):
        progress = (i + 1) / total_combinations
        progress_bar.progress(progress)
        status_text.text(f"Evaluating combination {i+1} of {total_combinations}...")
        
        w, pe, err = score_portfolio(combo, exposures, target)
        if err < best[2]:
            best = (combo, w, err)
    
    progress_bar.empty()
    status_text.empty()
    return best

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
    # Load data
    with st.spinner("Loading data..."):
        tickers, factors, R, F = load_data()
    
    if R is None:
        st.error("Failed to load data. Please check that all CSV files exist in the data/ directory.")
        return
    
    # Calculate exposures
    with st.spinner("Calculating factor exposures..."):
        exposures = calculate_exposures(R, F, tickers, factors)
    
    st.success(f"Loaded data for {len(tickers)} ETFs from {R.index[0].strftime('%Y-%m-%d')} to {R.index[-1].strftime('%Y-%m-%d')}")
    
    # Sidebar for inputs
    st.sidebar.header("Portfolio Configuration")
    
    # Factor weights input
    st.sidebar.subheader("Factor Weights")
    factor_names = list(factors.keys())
    factor_weights = {}
    
    # Default weights
    defaults = [0.4, 0.2, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0]
    
    for i, factor in enumerate(factor_names):
        factor_weights[factor] = st.sidebar.number_input(
            f"{factor.title()} Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=defaults[i], 
            step=0.05,
            key=f"weight_{factor}"
        )
    
    # Display total weight
    total_weight = sum(factor_weights.values())
    if abs(total_weight - 1.0) > 0.001:
        st.sidebar.error(f"⚠️ Total weight: {total_weight:.3f} (must equal 1.0)")
    else:
        st.sidebar.success(f"✅ Total weight: {total_weight:.3f}")
    
    # Number of ETFs
    n_etfs = st.sidebar.number_input(
        "Number of ETFs in Portfolio", 
        min_value=1, 
        max_value=len(tickers), 
        value=3, 
        step=1
    )
    
    # Optimization button
    if st.sidebar.button("🚀 Optimize Portfolio", type="primary"):
        if abs(total_weight - 1.0) > 0.001:
            st.error("Please ensure factor weights sum to 1.0 before optimizing.")
            return
        
        # Convert factor weights to array
        target = np.array([factor_weights[factor] for factor in factor_names])
        
        # Run optimization
        with st.spinner(f"Finding optimal {n_etfs}-ETF portfolio..."):
            combo, weights, error = find_best_subset(exposures, target, n_etfs)
        
        # Store results in session state
        st.session_state.optimization_results = {
            'combo': combo,
            'weights': weights,
            'error': error,
            'target': target,
            'factor_names': factor_names
        }
        
        st.success("Optimization completed!")
    
    # Display results if available
    if 'optimization_results' in st.session_state:
        results = st.session_state.optimization_results
        combo = results['combo']
        weights = results['weights']
        error = results['error']
        target = results['target']
        factor_names = results['factor_names']
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Portfolio Results", "📊 Performance Analysis", "🔬 Advanced Analytics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimized Portfolio")
                
                # Portfolio composition
                portfolio_df = pd.DataFrame({
                    'ETF': combo,
                    'Weight': weights,
                    'Weight (%)': weights * 100
                })
                st.dataframe(portfolio_df, hide_index=True)
                
                st.metric("Optimization Error", f"{error:.6f}")
                
                # Factor exposure comparison
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
                # Calculate portfolio returns
                port_rets = R[list(combo)].dot(weights)
                spy_rets = R["SPY"]
                
                # Performance metrics
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
            
            # Calculate cumulative returns
            initial_capital = 1000000
            cum_portfolio = (1 + port_rets).cumprod() * initial_capital
            cum_spy = (1 + spy_rets).cumprod() * initial_capital
            
            # Create performance plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(cum_portfolio.index, cum_portfolio.values, label=f"Optimized {len(combo)}-ETF Portfolio", linewidth=2)
            ax.plot(cum_spy.index, cum_spy.values, label="SPY (Benchmark)", linewidth=2)
            ax.set_title("Cumulative Portfolio Value", fontsize=14, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Performance metrics table
            st.subheader("Detailed Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Volatility',
                    'Sharpe Ratio',
                    'Maximum Drawdown',
                    'Beta (vs SPY)'
                ],
                'Portfolio': [
                    f"{metrics['portfolio']['total_return']:.2%}",
                    f"{metrics['portfolio']['annual_return']:.2%}",
                    f"{metrics['portfolio']['volatility']:.2%}",
                    f"{metrics['portfolio']['sharpe_ratio']:.3f}",
                    f"{metrics['portfolio']['max_drawdown']:.2%}",
                    f"{metrics['portfolio']['beta']:.3f}"
                ],
                'SPY Benchmark': [
                    f"{metrics['benchmark']['total_return']:.2%}",
                    f"{metrics['benchmark']['annual_return']:.2%}",
                    f"{metrics['benchmark']['volatility']:.2%}",
                    f"{metrics['benchmark']['sharpe_ratio']:.3f}",
                    f"{metrics['benchmark']['max_drawdown']:.2%}",
                    "1.000"
                ]
            })
            
            st.dataframe(metrics_df, hide_index=True)
            
            # Final values
            final_portfolio = cum_portfolio.iloc[-1]
            final_spy = cum_spy.iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Final Portfolio Value", 
                    f"${final_portfolio:,.2f}",
                    f"{(final_portfolio/initial_capital - 1)*100:.2f}%"
                )
            with col2:
                st.metric(
                    "Final SPY Value", 
                    f"${final_spy:,.2f}",
                    f"{(final_spy/initial_capital - 1)*100:.2f}%"
                )
        
        with tab3:
            st.subheader("Factor Exposure Matrix")
            st.dataframe(exposures.round(4))
            
            st.subheader("Correlation Matrix")
            # Ensure SPY is included without duplication
            etfs_for_corr = list(combo)
            if 'SPY' not in etfs_for_corr:
                etfs_for_corr.append('SPY')
            correlation_matrix = R[etfs_for_corr].corr()
            st.dataframe(correlation_matrix.round(3))
            
            st.subheader("Rolling Returns (30-day)")
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

if __name__ == "__main__":
    main() 