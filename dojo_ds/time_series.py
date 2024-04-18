import statsmodels.tsa.api as tsa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def get_adfuller_results(ts, alpha=.05, label='adfuller', **kwargs):
    """
    Uses statsmodels' adfuller function to test a univariate time series for stationarity.
    Null hypothesis: The time series is NOT stationary. (It "has a unit root".)
    Interpretation: A p-value less than alpha (.05) means the ts IS stationary.
    (We reject the null hypothesis that it is not stationary.)

    Parameters:
    - ts (array-like): The time series data.
    - alpha (float): The significance level used for interpreting p-value. Default is 0.05.
    - label (str): The label for the output DataFrame. Default is 'adfuller'.
    - **kwargs: Additional keyword arguments to be passed to statsmodels' adfuller function.

    Returns:
    - results (DataFrame): DataFrame with the following columns/results:
        - "Test Statistic": The adfuller test statistic.
        - "# of Lags Used": The number of lags used in the calculation.
        - "# of Observations": The number of observations used.
        - "p-value": The p-value for the hypothesis test.
        - "alpha": The significance level used for interpreting p-value.
        - "sig/stationary?": A simplified interpretation of the p-value.

    ADFULLER DOCUMENTATION:
    For the full adfuller documentation, see:
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
    """
    (test_stat, pval, nlags, nobs, crit_vals_d, icbest) = tsa.adfuller(ts, **kwargs)
    adfuller_results = {
        'Test Statistic': test_stat,
        '# of Lags Used': nlags,
        '# of Observations': nobs,
        'p-value': round(pval, 6),
        'alpha': alpha,
        'sig/stationary?': pval < alpha
    }
    return pd.DataFrame(adfuller_results, index=[label])


def get_sig_lags(ts, type='ACF', nlags=None, alpha=0.5):
    """
    Calculates the significant lags for the autocorrelation function (ACF) or partial autocorrelation function (PACF) of a time series.

    Parameters:
    - ts (array-like): The time series data.
    - type (str): The type of lag to calculate. Must be either 'ACF' or 'PACF'. Default is 'ACF'.
    - nlags (int): The number of lags to calculate. Default is None.
    - alpha (float): The significance level for determining significant lags. Default is 0.5.

    Returns:
    - sig_lags (array-like): The significant lags.

    Raises:
    - Exception: If the type is not 'ACF' or 'PACF'.
    """
    if type == 'ACF':
        corr_values, conf_int = tsa.stattools.acf(ts, alpha=alpha, nlags=nlags)
    elif type == 'PACF':
        corr_values, conf_int = tsa.stattools.pacf(ts, alpha=alpha, nlags=nlags)
    else:
        raise Exception("type must be either 'ACF' or 'PACF'")

    lags = range(len(corr_values))
    corr_df = pd.DataFrame({
        type: corr_values,
        'Lags': lags,
        'lower ci': conf_int[:, 0] - corr_values,
        'upper ci': conf_int[:, 1] - corr_values
    })
    corr_df = corr_df.set_index("Lags")

    filter_sig_lags = (corr_df[type] < corr_df['lower ci']) | (corr_df[type] > corr_df['upper ci'])
    sig_lags = corr_df.index[filter_sig_lags]
    sig_lags = sig_lags[sig_lags != 0]

    return sig_lags


def plot_acf_pacf(ts, nlags=40, figsize=(10, 5), annotate_sig=False, alpha=.05, acf_kws={}, pacf_kws={}, annotate_seas=False, m=None, seas_color='black'):
    """
    Plots the autocorrelation function (ACF) and partial autocorrelation function (PACF) of a time series.

    Parameters:
    - ts (array-like): The time series data.
    - nlags (int): The number of lags to include in the plot. Default is 40.
    - figsize (tuple): The size of the figure. Default is (10, 5).
    - annotate_sig (bool): Whether to annotate significant lags. Default is False.
    - alpha (float): The significance level for determining significant lags. Default is 0.05.
    - acf_kws (dict): Additional keyword arguments to be passed to statsmodels' plot_acf function.
    - pacf_kws (dict): Additional keyword arguments to be passed to statsmodels' plot_pacf function.
    - annotate_seas (bool): Whether to annotate seasonal lags. Default is False.
    - m (int): The number of observations per season. Required if annotate_seas is True.
    - seas_color (str): The color of the seasonal lines. Default is 'black'.

    Returns:
    - fig (Figure): The matplotlib Figure object.
    """
    fig, axes = plt.subplots(nrows=2, figsize=figsize)
    sig_vline_kwargs = dict(ls=':', lw=1, zorder=0, color='red')

    tsa.graphics.plot_acf(ts, ax=axes[0], lags=nlags, **acf_kws)

    if annotate_sig:
        sig_acf_lags = get_sig_lags(ts, nlags=nlags, alpha=alpha, type='ACF')
        for lag in sig_acf_lags:
            axes[0].axvline(lag, label='sig', **sig_vline_kwargs)

    tsa.graphics.plot_pacf(ts, ax=axes[1], lags=nlags, **pacf_kws)

    if annotate_sig:
        sig_pacf_lags = get_sig_lags(ts, nlags=nlags, alpha=alpha, type='PACF')
        for lag in sig_pacf_lags:
            axes[1].axvline(lag, label='sig', **sig_vline_kwargs)

    if annotate_seas:
        if m is None:
            raise Exception("Must define value of m if annotate_seas=True.")

        n_seasons = nlags // m
        seas_vline_kwargs = dict(ls='--', lw=1, alpha=.7, color=seas_color, zorder=-1)

        for i in range(1, n_seasons + 1):
            axes[0].axvline(m * i, **seas_vline_kwargs, label="season")
            axes[1].axvline(m * i, **seas_vline_kwargs, label="season")

    fig.tight_layout()
    return fig


def regression_metrics_ts(ts_true, ts_pred, label="", verbose=True, output_dict=False):
    """
    Calculates regression metrics for comparing true and predicted time series data.

    Parameters:
    - ts_true (array-like): The true time series data.
    - ts_pred (array-like): The predicted time series data.
    - label (str): The label for the metrics. Default is an empty string.
    - verbose (bool): Whether to print the metrics. Default is True.
    - output_dict (bool): Whether to return the metrics as a dictionary. Default is False.

    Returns:
    - metrics (dict): The regression metrics as a dictionary. Only returned if output_dict is True.
    """
    mae = mean_absolute_error(ts_true, ts_pred)
    mse = mean_squared_error(ts_true, ts_pred)
    rmse = mean_squared_error(ts_true, ts_pred, squared=False)
    r_squared = r2_score(ts_true, ts_pred)
    mae_perc = mean_absolute_percentage_error(ts_true, ts_pred) * 100

    if verbose:
        header = "---" * 20
        print(header, f"Regression Metrics: {label}", header, sep="\n")
        print(f"- MAE = {mae:,.3f}")
        print(f"- MSE = {mse:,.3f}")
        print(f"- RMSE = {rmse:,.3f}")
        print(f"- R^2 = {r_squared:,.3f}")
        print(f"- MAPE = {mae_perc:,.2f}%")

    if output_dict:
        metrics = {
            "Label": label,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R^2": r_squared,
            "MAPE(%)": mae_perc,
        }
        return metrics


def plot_forecast(ts_train, ts_test, forecast_df, n_train_lags=None, 
                  figsize=(10,4), title='Comparing Forecast vs. True Data'):
    """
    Plots the training data, test data, and forecast with confidence intervals.

    Parameters:
    ts_train (pandas.Series): Time series data for training.
    ts_test (pandas.Series): Time series data for testing.
    forecast_df (pandas.DataFrame): DataFrame containing forecasted values and confidence intervals.
    n_train_lags (int, optional): Number of training lags to plot. If not specified, all training data will be plotted.
    figsize (tuple, optional): Figure size. Default is (10, 4).
    title (str, optional): Title of the plot. Default is 'Comparing Forecast vs. True Data'.

    Returns:
    fig (matplotlib.figure.Figure): The generated figure.
    ax (matplotlib.axes.Axes): The axes of the generated figure.
    """
    ### PLot training data, and forecast (with upper/,lower ci)
    fig, ax = plt.subplots(figsize=figsize)

    # setting the number of train lags to plot if not specified
    if n_train_lags==None:
        n_train_lags = len(ts_train)
            
    # Plotting Training  and test data
    ts_train.iloc[-n_train_lags:].plot(ax=ax, label="train")
    ts_test.plot(label="test", ax=ax)

    # Plot forecast
    forecast_df['mean'].plot(ax=ax, color='green', label="forecast")

    # Add the shaded confidence interval
    ax.fill_between(forecast_df.index, 
                    forecast_df['mean_ci_lower'],
                   forecast_df['mean_ci_upper'],
                   color='green', alpha=0.3,  lw=2)

    # set the title and add legend
    ax.set_title(title)
    ax.legend();
    
    return fig, ax

