import matplotlib.dates as mdates


def set_date_ticks(ax, major=None, minor=None):
    '''
    Applies major/minor locators to a date x-axis
    so minor ticks stay aligned with major ticks
    '''
    ax.xaxis.set_major_locator(major or mdates.YearLocator())
    ax.xaxis.set_minor_locator(minor or mdates.MonthLocator())
