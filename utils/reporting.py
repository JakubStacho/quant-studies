import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from metrics import (get_compound_returns,
                      get_drawdown,
                      get_cagr,
                      get_annualized_volatility,
                      get_annualized_sharpe_series,
                      get_annualized_sharpe_rolling,
                      )
from plotting import set_date_ticks


def get_summary_stats(returns_df, risk_free_returns, vals_per_year=12):
    '''
    Returns a dataframe of summary stats (CAGR,
    Max Drawdown, and annualized Sharpe ratio)
    for each column in the given returns df
    '''
    compound_returns = get_compound_returns(returns_df)
    drawdown         = get_drawdown(returns_df)
    excess_returns   = returns_df - risk_free_returns.values.reshape(-1, 1)

    stats = pd.DataFrame({'CAGR [%]'         : get_cagr(compound_returns) * 100,
                          'Volatility [%]'   : get_annualized_volatility(returns_df, vals_per_year) * 100,
                          'Max Drawdown [%]' : drawdown.min() * 100,
                          'Sharpe'           : excess_returns.apply(get_annualized_sharpe_series,
                                                                    vals_per_year=vals_per_year),
                          }).T

    return stats


def print_summary_stats(stats):
    '''
    Prints a dataframe of summary stats (as returned by
    get_summary_stats) as a nicely aligned table, with
    tickers as rows and stat names as columns. Each
    column is sized to its own content, with a single
    space of padding on either side.
    '''
    stats = stats.T

    label_width = max(len(label) for label in stats.index)
    value_strs  = stats.map(lambda v: f'{v:.2f}')
    col_widths  = {col: max(len(col), value_strs[col].str.len().max()) for col in stats.columns}

    header = f'{"":<{label_width}} ' + ''.join(f' {col:^{col_widths[col]}} ' for col in stats.columns)
    print(header)
    print('-' * len(header))

    for label, row in value_strs.iterrows():
        values = ''.join(f' {row[col]:^{col_widths[col]}} ' for col in stats.columns)
        print(f'{label:<{label_width}} {values}')


def performance_report(returns_df, risk_free_returns, vals_per_year=12):
    '''
    Run a quick performance report to compare
    all the columns in returns_df. This prints
    a few relevant metrics and generates a plot.
    Returns a dictionary of the rolling metrics
    '''
    # print summary stats
    summary_stats = get_summary_stats(returns_df, risk_free_returns, vals_per_year)
    print_summary_stats(summary_stats)

    # calculate and plot rolling metrics
    excess_returns_df = returns_df - risk_free_returns.values.reshape(-1,1)

    compound_returns = get_compound_returns(returns_df)
    rolling_drawdown = get_drawdown(returns_df)
    rolling_sharpe   = get_annualized_sharpe_rolling(excess_returns_df, window=36, vals_per_year=vals_per_year)

    fig, axs = plt.subplots(nrows=3, figsize=(9, 6), sharex=True)

    for ticker in returns_df.columns:
        # cumulative compound return
        ax = axs[0]
        ax.plot(compound_returns[ticker] * 100., label=ticker)
        ax.set_ylabel('Return [%]')

        # rolling drawdown
        ax = axs[1]
        ax.plot(rolling_drawdown[ticker] * 100., label=ticker)
        ax.set_ylabel('Drawdown [%]')

        # rolling sharpe
        ax = axs[2]
        ax.plot(rolling_sharpe[ticker], label=ticker)
        ax.set_ylabel('Rolling 3 Year Sharpe')

    axs[0].legend(loc            = 'upper right',
                bbox_to_anchor = (1, 1.2),
                ncols          = 5,
                frameon        = False,
                )

    for ax in axs:
        set_date_ticks(ax)
        ax.yaxis.set_minor_locator(AutoMinorLocator()) 
        ax.tick_params(top=True, bottom=True, left=True, right=True)
        ax.grid(axis='y', linewidth=0.75, alpha=0.5)

    fig.tight_layout()
    # plt.show()

    return_dict = {'compound_returns' : compound_returns,
                   'drawdown'         : rolling_drawdown,
                   'sharpe'           : rolling_sharpe,
                   }

    return return_dict, fig, ax
