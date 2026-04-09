import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# ================== 1. Portfolio Setup (FRTB-style Trading Desk) ==================
tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'TSLA']   # Example desk
weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])

# FRTB Liquidity Horizon Buckets (in days) - official BCBS mapping example
liquidity_horizons = {
    'AAPL': 10,   # Large-cap equity → Bucket 1
    'MSFT': 10,
    'GOOGL': 10,
    'JPM': 20,    # Bank equity → Bucket 2
    'TSLA': 40    # High-volatility equity → Bucket 3
}

data = yf.download(tickers, start='2024-04-09', end='2026-04-09',progress=False)['Close']
returns = data.pct_change().dropna()

portfolio_returns = returns.dot(weights)   # Daily portfolio returns

print(f"Portfolio data: {len(portfolio_returns)} days | From {portfolio_returns.index[0].date()}")

# ================== 2. FRTB Liquidity-Adjusted Expected Shortfall ==================
def frtb_liquidity_adjusted_es(returns_series, window=252, confidence=0.975):
    """Shortcut FRTB IMA: Historical ES + Liquidity Horizon Scaling"""
    es_series = pd.Series(index=returns_series.index, dtype=float)
    var_series = pd.Series(index=returns_series.index, dtype=float)  # for backtesting
    
    for i in range(window, len(returns_series)):
        hist_returns = returns_series.iloc[i-window:i]
        
        # Step 1: Base 1-day ES (97.5%)
        losses = -hist_returns.values
        var_1d = np.percentile(losses, (1 - confidence) * 100)
        es_1d = losses[losses >= var_1d].mean() if len(losses[losses >= var_1d]) > 0 else var_1d
        
        # Step 2: Liquidity Horizon Scaling (FRTB official nested formula - simplified)
        # We scale each asset's contribution by sqrt(LH / 10)
        scaled_losses = np.zeros(len(hist_returns))
        for asset, w in zip(returns.columns, weights):
            lh = liquidity_horizons.get(asset, 10)          # default 10 days
            asset_returns = returns[asset].iloc[i-window:i]
            scaled = asset_returns * w * np.sqrt(lh / 10.0)   # FRTB scaling factor
            scaled_losses += -scaled.values
        
        # Final liquidity-adjusted ES
        es_liquidity = scaled_losses[scaled_losses >= np.percentile(scaled_losses, (1-confidence)*100)].mean()
        
        es_series.iloc[i] = es_liquidity
        var_series.iloc[i] = var_1d * np.sqrt(10)  # 10-day VaR for reference
    
    return es_series, var_series

# Calculate
es_97_5, var_10d = frtb_liquidity_adjusted_es(portfolio_returns, window=252)

# ================== 3. Plot Results ==================
plt.figure(figsize=(14, 7))
plt.plot(portfolio_returns.index, -portfolio_returns * 100, label='Actual Daily Loss (%)', color='gray', alpha=0.6)
plt.plot(es_97_5.index, es_97_5 * 100, label='FRTB 97.5% Liquidity-Adjusted ES (%)', color='red', linewidth=2)
plt.title('Shortcut FRTB IMA: Liquidity-Adjusted Expected Shortfall (97.5%)')
plt.ylabel('Loss (%)')
plt.legend()
plt.grid(True)
plt.show()

# ================== 4. Simple Backtesting (Basel-style + ES check) ==================
def frtb_backtest(es_series, actual_returns, window=252):
    test_es = es_series.dropna()
    test_actual = -actual_returns[-len(test_es):]
    
    exceptions = (test_actual > test_es).sum()
    n = len(test_es)
    expected_exceptions = (1 - 0.975) * n
    
    print("\n=== FRTB IMA Backtesting Results ===")
    print(f"Observations tested     : {n}")
    print(f"Exceptions (breaches)   : {exceptions} (expected ≈ {expected_exceptions:.1f})")
    print(f"Exception rate          : {exceptions/n*100:.2f}%")
    print("Zone (simplified):", "GREEN" if exceptions <= 0.05*n else "YELLOW" if exceptions <= 0.10*n else "RED")

frtb_backtest(es_97_5, portfolio_returns)

# ================== 5. Sensitivity Analysis ==================
print("\nSensitivity to Liquidity Horizons:")
for factor in [1.0, 1.5, 2.0]:
    # Quick re-run with multiplied horizons
    temp_es = frtb_liquidity_adjusted_es(portfolio_returns, window=252)[0].dropna().mean() * factor
    print(f"Liquidity multiplier ×{factor} → Avg ES = {temp_es*100:.2f}%")
