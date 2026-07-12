# Portfolio Optimization

Mean-variance optimization applied to a universe of retail-accessible ETFs covering 20+ years of monthly returns (2005–2026).

## Notebooks

- **`pull-data.ipynb`** — downloads ETF prices from `yfinance` and the risk-free rate from the Kenneth French Data Library. Saves data to `data/`
- **`mean-variance.ipynb`** — the main analysis: correlation analysis, optimization, efficient frontier, and backtest

## What's Covered

**Asset universe:** SPY (US large cap), VNQ (REITs), GLD (gold), LQD (investment grade bonds), TLT (long-term Treasuries)

**Covariance estimation**
Daily returns are used to estimate the covariance matrix even though the optimization runs on monthly returns — more observations means tighter standard errors on correlation estimates. Ledoit-Wolf shrinkage is applied as a robustness check against a noisy sample covariance.

**Optimization**
Three portfolios are solved following the MOSEK Portfolio Optimization Cookbook:
- Global minimum variance
- Maximum Sharpe ratio (tangency portfolio)
- Full efficient frontier (sweeping target return levels)

Results are plotted against a cloud of random long-only portfolios and a hand-built 60/40 SPY/LQD benchmark, with the Capital Market Line anchored at the Fama-French risk-free rate.

**Correlation regime analysis**
Rolling correlations between each asset and SPY are plotted against SPY's drawdown to show how diversification breaks down in a crisis. A notable regime shift is visible from 2022 onward: long-duration bonds (TLT, LQD) flipped from negatively correlated with equities to positively correlated as inflation risk replaced growth risk as the dominant macro driver.

**Walk-forward backtest**
Monthly rebalancing backtest to check whether the optimization adds out-of-sample value versus a passive SPY allocation.

## Key Observations

The standard error of a mean return estimate over 20 years of monthly data is often as large as the spread between assets' mean returns — the optimizer is partly chasing noise, not real differences in expected return. And the covariance matrix clearly isn't stationary, which is a core MVO assumption. Both are well-known problems in practice; the exploration here makes them concrete and visible in the data.
