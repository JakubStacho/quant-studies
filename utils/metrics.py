import numpy as np


def get_compound_returns(returns_df):
    '''
    Returns a dataframe of compound returns
    for each column in the given returns df
    '''
    return (returns_df + 1).cumprod(axis=0) - 1


def get_annualized_sharpe_series(excess_returns_series, vals_per_year=12):
    '''
    Returns the annualized sharpe ratio
    for the given series of excess returns
    '''
    mean_return = np.mean(excess_returns_series)
    stdv_return = np.std(excess_returns_series)

    return np.sqrt(vals_per_year) * mean_return / stdv_return


def get_annualized_sharpe_rolling(excess_returns_df, window=36, vals_per_year=12):
    '''
    Returns a dataframe of the rolling
    annualized Sharpe ratio for each
    column in the given excess returns df
    '''
    return excess_returns_df.rolling(window=window).apply(get_annualized_sharpe_series,
                                                          kwargs = {'vals_per_year': vals_per_year},
                                                          raw    = True)


def get_rolling_correlation(returns_df, reference_series, window=36):
    '''
    Returns a dataframe of the rolling correlation between
    each column in returns_df and a reference series
    '''
    return returns_df.rolling(window=window).corr(reference_series)


def get_annualized_volatility(returns_df, vals_per_year=12):
    '''
    Returns the annualized volatility (standard
    deviation of returns) for each column in
    the given returns df
    '''
    return returns_df.std() * np.sqrt(vals_per_year)


def get_annualized_mean_return(returns_df, vals_per_year=12):
    '''
    Returns the annualized arithmetic mean return for
    each column in the given returns df
    '''
    return returns_df.mean() * vals_per_year


def get_drawdown(returns_df):
    '''
    Returns a dataframe of the drawdown
    time series for each column in the
    given returns df
    '''
    running_value = (returns_df + 1).cumprod(axis=0)
    running_max   = running_value.cummax(axis=0)

    drawdown  = ((running_value / running_max) - 1)

    return drawdown


def get_cagr(compound_returns_df):
    '''
    Returns the CAGR for each column in
    the given dataframe of compound returns
    '''
    start_date = compound_returns_df.index[0]
    end_date   = compound_returns_df.index[-1]
    n_years    = ((end_date - start_date).days / 365.2425) + (1./12.) # 365.2425 days per year, add extra month because data is month-end

    cagr = (compound_returns_df.iloc[-1] + 1)**(1 / n_years) - 1

    return cagr
