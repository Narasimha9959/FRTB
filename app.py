import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path
import appdirs as ad

# ====================== FIX FOR YFINANCE ON STREAMLIT CLOUD ======================
ad.user_cache_dir = lambda *args: "/tmp"
Path("/tmp").mkdir(exist_ok=True)
# ===============================================================================

st.set_page_config(page_title="FRTB Dashboard", layout="wide")

st.title("🚀 FRTB IMA Dashboard")
st.markdown("**Liquidity-Adjusted Expected Shortfall (97.5%)**")

# Sidebar
with st.sidebar:
    st.header("Portfolio Settings")
    
    tickers_input = st.text_input("Tickers", value="AAPL, MSFT, GOOGL, JPM, TSLA")
    weights_input = st.text_input("Weights", value="0.25, 0.25, 0.20, 0.15, 0.15")
    window = st.slider("Lookback Window (days)", 126, 504, 252)
    confidence = st.selectbox("Confidence Level", [0.95, 0.975, 0.99], index=1)
    lh_mult = st.slider("Liquidity Horizon Multiplier", 0.5, 3.0, 1.0, 0.1)
    
    if st.button("Update Dashboard", type="primary"):
        st.session_state.update_clicked = True

# Calculation Function
def calculate_frtb_es(returns, weights, lh_dict, window=252, confidence=0.975, lh_mult=1.0):
    es_list = []
    for i in range(window, len(returns)):
        hist = returns.iloc[i-window:i]
        scaled_losses = np.zeros(len(hist))
        for j, asset in enumerate(returns.columns):
            lh = lh_dict.get(asset, 10) * lh_mult
            scaled_losses += -hist[asset].values * weights[j] * np.sqrt(lh / 10.0)
        
        threshold = np.percentile(scaled_losses, (1 - confidence) * 100)
        tail = scaled_losses[scaled_losses >= threshold]
        es_val = tail.mean() if len(tail) > 0 else threshold
        es_list.append(es_val)
    return pd.Series(es_list, index=returns.index[window:])

# Main Logic
if st.session_state.get("update_clicked", False):
    try:
        tickers = [t.strip().upper() for t in tickers_input.split(',')]
        weights = np.array([float(w.strip()) for w in weights_input.split(',')])
        
        if len(tickers) != len(weights) or abs(sum(weights) - 1) > 0.02:
            st.error("Weights must sum to 1.0")
            st.stop()

        data = yf.download(tickers, start='2020-01-01', progress=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        
        returns = data.pct_change().dropna()
        portfolio_returns = returns.dot(weights)

        lh_dict = {t: 10 if idx < 3 else 20 if idx < 4 else 40 for idx, t in enumerate(tickers)}

        es_series = calculate_frtb_es(returns, weights, lh_dict, window, confidence, lh_mult)

        # Charts & Metrics
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio_returns.index, y=-portfolio_returns*100, name="Actual Loss (%)", line=dict(color="gray")))
            fig.add_trace(go.Scatter(x=es_series.index, y=es_series*100, name=f"{confidence*100:.1f}% FRTB ES", line=dict(color="red", width=3)))
            fig.update_layout(template="plotly_dark", height=600, xaxis_title="Date", yaxis_title="Loss (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            exceptions = (-portfolio_returns[-len(es_series):] > es_series).sum()
            rate = exceptions / len(es_series) * 100 if len(es_series) > 0 else 0
            zone = "🟢 GREEN" if rate <= 5 else "🟡 YELLOW" if rate <= 10 else "🔴 RED"
            
            st.metric("Observations", len(es_series))
            st.metric("Exceptions", f"{exceptions} ({rate:.2f}%)")
            st.metric("Zone", zone)
            st.metric("Avg ES", f"{es_series.mean()*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Configure settings in sidebar and click **Update Dashboard**")

st.caption("FRTB Dashboard | Fixed for Streamlit Cloud")
