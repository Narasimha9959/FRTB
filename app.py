import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path
import appdirs as ad
from datetime import datetime

# ====================== IMPORTANT FIX FOR STREAMLIT CLOUD ======================
# This prevents yfinance cache errors on Streamlit Cloud
ad.user_cache_dir = lambda *args: "/tmp"
Path("/tmp").mkdir(exist_ok=True)
# ===============================================================================

st.set_page_config(
    page_title="FRTB IMA Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 FRTB Internal Models Approach (IMA) Dashboard")
st.markdown("**Liquidity-Adjusted Expected Shortfall (97.5%)** | Historical Simulation Shortcut")

# ================== Sidebar Controls ==================
with st.sidebar:
    st.header("Portfolio Configuration")
    
    tickers_input = st.text_input(
        "Tickers (comma separated)",
        value="AAPL, MSFT, GOOGL, JPM, TSLA"
    )
    
    weights_input = st.text_input(
        "Weights (comma separated - must sum to 1)",
        value="0.25, 0.25, 0.20, 0.15, 0.15"
    )
    
    window = st.slider(
        "Lookback Window (days)", 
        min_value=126, 
        max_value=504, 
        value=252, 
        step=1
    )
    
    confidence = st.selectbox(
        "Confidence Level",
        options=[0.95, 0.975, 0.99],
        format_func=lambda x: f"{x*100:.1f}%",
        index=1
    )
    
    lh_multiplier = st.slider(
        "Liquidity Horizon Multiplier", 
        min_value=0.5, 
        max_value=3.0, 
        value=1.0, 
        step=0.1
    )
    
    update_button = st.button("🔄 Update Dashboard", type="primary")

# ================== FRTB Calculation Function ==================
def calculate_frtb_es(returns, weights, lh_dict, window=252, confidence=0.975, lh_mult=1.0):
    es_list = []
    dates = []
    
    for i in range(window, len(returns)):
        hist = returns.iloc[i - window:i]
        dates.append(returns.index[i])
        
        # Liquidity-adjusted scaled losses
        scaled_losses = np.zeros(len(hist))
        for j, asset in enumerate(returns.columns):
            lh = lh_dict.get(asset, 10) * lh_mult
            scaled_losses += -hist[asset].values * weights[j] * np.sqrt(lh / 10.0)
        
        # Calculate Expected Shortfall
        threshold = np.percentile(scaled_losses, (1 - confidence) * 100)
        tail_losses = scaled_losses[scaled_losses >= threshold]
        es_val = tail_losses.mean() if len(tail_losses) > 0 else threshold
        
        es_list.append(es_val)
    
    return pd.Series(es_list, index=dates)

# ================== Main App Logic ==================
if update_button:
    try:
        # Parse inputs
        tickers = [t.strip().upper() for t in tickers_input.split(',')]
        weights = np.array([float(w.strip()) for w in weights_input.split(',')])
        
        if len(tickers) != len(weights):
            st.error("❌ Number of tickers and weights must be the same!")
            st.stop()
        
        if abs(weights.sum() - 1.0) > 0.02:
            st.warning("⚠️ Weights should sum to approximately 1.0")

        # Download data
        with st.spinner("Downloading market data..."):
            data = yf.download(tickers, start='2020-01-01', progress=False)['Adj Close']
        
        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        
        returns = data.pct_change().dropna()
        portfolio_returns = returns.dot(weights)

        # Define Liquidity Horizons
        lh_dict = {ticker: 10 if i < 3 else 20 if i < 4 else 40 
                  for i, ticker in enumerate(tickers)}

        # Calculate FRTB ES
        es_series = calculate_frtb_es(returns, weights, lh_dict, window, confidence, lh_multiplier)

        # ================== Display Results ==================
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("FRTB Liquidity-Adjusted Expected Shortfall vs Actual Losses")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_returns.index, 
                y=-portfolio_returns * 100,
                name='Actual Daily Loss (%)',
                line=dict(color='lightgray', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=es_series.index, 
                y=es_series * 100,
                name=f'{confidence*100:.1f}% FRTB ES',
                line=dict(color='red', width=3)
            ))
            fig.update_layout(
                height=650,
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Loss (%)",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Backtesting Summary")
            test_es = es_series.dropna()
            test_actual = -portfolio_returns[-len(test_es):]
            exceptions = (test_actual > test_es).sum()
            n = len(test_es)
            exception_rate = (exceptions / n * 100) if n > 0 else 0
            
            zone_color = "🟢 GREEN" if exception_rate <= 5 else "🟡 YELLOW" if exception_rate <= 10 else "🔴 RED"
            
            st.metric("Observations", f"{n:,}")
            st.metric("Exceptions", f"{exceptions} ({exception_rate:.2f}%)")
            st.metric("Backtesting Zone", zone_color)
            st.metric("Average ES", f"{es_series.mean()*100:.2f}%")
            st.metric("Maximum ES", f"{es_series.max()*100:.2f}%")

        # Sensitivity Analysis
        st.subheader("Sensitivity to Liquidity Horizon Multiplier")
        multipliers = [0.5, 1.0, 1.5, 2.0, 2.5]
        avg_es_values = []
        
        for m in multipliers:
            temp_es = calculate_frtb_es(returns, weights, lh_dict, window, confidence, m)
            avg_es_values.append(temp_es.mean() * 100)
        
        sens_fig = go.Figure()
        sens_fig.add_trace(go.Bar(
            x=[f"{m}x" for m in multipliers],
            y=avg_es_values,
            marker_color='royalblue'
        ))
        sens_fig.update_layout(
            height=400,
            template="plotly_dark",
            xaxis_title="Liquidity Horizon Multiplier",
            yaxis_title="Average ES (%)"
        )
        st.plotly_chart(sens_fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Tip: Make sure tickers are valid and weights sum to 1.0")

else:
    st.info("👈 Please configure your portfolio in the sidebar and click **Update Dashboard**")

st.caption("FRTB Shortcut Dashboard | Liquidity-Adjusted Expected Shortfall (97.5%) | Fixed for Streamlit Cloud")
