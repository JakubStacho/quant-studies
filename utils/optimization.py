import numpy as np
import pandas as pd
from scipy.optimize import minimize


def get_portfolio_stats(weights, mean_returns, cov_matrix):
    '''
    Returns the (annualized return, annualized volatility)
    for a portfolio with the given weights, mean returns,
    and covariance matrix
    '''
    weights      = np.asarray(weights)
    mean_returns = np.asarray(mean_returns)
    cov_matrix   = np.asarray(cov_matrix)

    portfolio_return = weights @ mean_returns
    portfolio_vol    = np.sqrt(weights @ cov_matrix @ weights)

    return portfolio_return, portfolio_vol


def solve_min_variance(mean_returns, cov_matrix, target_return=None):
    '''
    Solves for the long-only minimum-variance portfolio weights
    (weights sum to 1, each between 0 and 1). If target_return is
    given, adds an equality constraint so this solves one point on
    the efficient frontier; if None, solves the Global Minimum
    Variance portfolio
    '''
    n_assets = len(mean_returns)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda w: w @ mean_returns.values - target_return})

    bounds = [(0, 1)] * n_assets
    x0     = np.repeat(1 / n_assets, n_assets)

    result = minimize(lambda w: w @ cov_matrix.values @ w,
                       x0,
                       method      = 'SLSQP',
                       bounds      = bounds,
                       constraints = constraints,
                       )

    if not result.success:
        raise RuntimeError(f'Min variance solve failed: {result.message}')

    return pd.Series(result.x, index=mean_returns.index)


def solve_max_sharpe(mean_returns, cov_matrix, risk_free_rate):
    '''
    Solves for the long-only max-Sharpe-ratio (tangency) portfolio
    weights (weights sum to 1, each between 0 and 1)
    '''
    n_assets = len(mean_returns)

    def negative_sharpe(w):
        portfolio_return, portfolio_vol = get_portfolio_stats(w, mean_returns, cov_matrix)
        return -(portfolio_return - risk_free_rate) / portfolio_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n_assets
    x0          = np.repeat(1 / n_assets, n_assets)

    result = minimize(negative_sharpe,
                       x0,
                       method      = 'SLSQP',
                       bounds      = bounds,
                       constraints = constraints,
                       )

    if not result.success:
        raise RuntimeError(f'Max Sharpe solve failed: {result.message}')

    return pd.Series(result.x, index=mean_returns.index)


def build_efficient_frontier(mean_returns, cov_matrix, n_points=50):
    '''
    Sweeps target returns from the min to max asset return and
    solves the minimum-variance portfolio at each, returning a
    dataframe of target_return, volatility, and per-asset weights
    for each frontier point that converged
    '''
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)

    rows = []
    for target_return in target_returns:
        try:
            weights = solve_min_variance(mean_returns, cov_matrix, target_return=target_return)
        except RuntimeError:
            continue

        _, volatility = get_portfolio_stats(weights, mean_returns, cov_matrix)

        row                  = weights.to_dict()
        row['target_return'] = target_return
        row['volatility']    = volatility
        rows.append(row)

    return pd.DataFrame(rows)
