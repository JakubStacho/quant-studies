import matplotlib.dates as mdates
import matplotlib.ticker as mticker


def set_date_ticks(ax, major=None, minor=None, label_every=2):
    '''
    Applies major/minor locators to a date x-axis
    '''
    ax.xaxis.set_major_locator(major or mdates.YearLocator(1))
    ax.xaxis.set_minor_locator(minor or mdates.MonthLocator(bymonth=[1, 7]))

    if label_every > 1:
        def _year_label(x, _pos):
            year = mdates.num2date(x).year
            return str(year) if year % label_every == 0 else ''

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_year_label))
