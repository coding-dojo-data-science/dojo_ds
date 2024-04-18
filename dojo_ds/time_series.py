import statsmodels.tsa.api as tsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.api as tsa
import numpy as np
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


def thiels_U(ts_true, ts_pred):
    """Calculate's Thiel's U metric for forecasting accuracy.
    Accepts true values and predicted values.
    Original Formula Source: https://docs.oracle.com/cd/E57185_01/CBREG/ch06s02s03s04.html
    Adapted Function from Source: https://github.com/jirvingphd/predicting-the-SP500-using-trumps-tweets_capstone-project/blob/cf11f6ed88721433d2c00cb1f8486206ab179cc0/bsds/my_keras_functions.py#L735
    Returns: 
        U (float)
        
    Thiel's U Value Interpretation:
    - <1  = Forecasting is better than guessing 
    - 1   = Forecasting is about as good as guessing
    - >1  = Forecasting is worse than guessing 
    """
    import numpy as np
    # sum_list = []
    num_list=[]
    denom_list=[]
    
    for t in range(len(ts_true)-1):
        
        num_exp = (ts_pred[t+1] - ts_true[t+1])/ts_true[t]
        num_list.append([num_exp**2])
        
        denom_exp = (ts_true[t+1] - ts_true[t])/ts_true[t]
        denom_list.append([denom_exp**2])
        
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U


def calc_thiels_U(*args, **kwargs):
    return thiels_U(*args, **kwargs)


def regression_metrics_ts(ts_true, ts_pred, label="", verbose=True, output_dict=False,
                          thiels_U=False):
    """
    Calculates regression metrics for comparing true and predicted time series data.

    Parameters:
    - ts_true (array-like): The true time series data.
    - ts_pred (array-like): The predicted time series data.
    - label (str): The label for the metrics. Default is an empty string.
    - verbose (bool): Whether to print the metrics. Default is True.
    - output_dict (bool): Whether to return the metrics as a dictionary. Default is False.
    - thiels_U (bool): Whether to calculate Thiel's U metric. Default is False.
                        - See docstring for thiels_U function for interpretation.

    Returns:
    - metrics (dict): The regression metrics as a dictionary. Only returned if output_dict is True.
    """
    mae = mean_absolute_error(ts_true, ts_pred)
    mse = mean_squared_error(ts_true, ts_pred)
    rmse = mean_squared_error(ts_true, ts_pred, squared=False)
    r_squared = r2_score(ts_true, ts_pred)
    mae_perc = mean_absolute_percentage_error(ts_true, ts_pred) * 100
    if thiels_U==True:
        U = calc_thiels_U(ts_true, ts_pred)
    if verbose:
        header = "---" * 20
        print(header, f"Regression Metrics: {label}", header, sep="\n")
        print(f"- MAE = {mae:,.3f}")
        print(f"- MSE = {mse:,.3f}")
        print(f"- RMSE = {rmse:,.3f}")
        print(f"- R^2 = {r_squared:,.3f}")
        print(f"- MAPE = {mae_perc:,.2f}%")
        
        if thiels_U:
            print(f"- Thiel's U = {U:,.2f}")
        print(header)
            
    if output_dict:
        metrics = {
            "Label": label,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R^2": r_squared,
            "MAPE(%)": mae_perc,
        }
        if thiels_U:
            metrics['Thiel\'s U'] = U
        return metrics





def plot_acf(ts, nlags=None, alpha=0.05, figsize=(12,4), annotate_sig=False): 
    """
    Plots the autocorrelation function (ACF) for a given time series.

    Parameters:
    - ts: The time series data.
    - nlags: The number of lags to include in the plot. If None, all lags are included.
    - alpha: The significance level for determining significant lags.
    - figsize: The size of the figure (width, height) in inches.
    - annotate_sig: Whether to annotate significant lags with red dashed lines.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=figsize) 
    tsa.graphics.plot_acf(ts, ax=ax, lags=nlags, alpha=alpha)

    if annotate_sig == True:
        sig_lags = get_sig_lags(ts,nlags=nlags,alpha=alpha, type='ACF')
        
        ## ADDING ANNOTATING SIG LAGS
        for lag in sig_lags:
            ax.axvline(lag, ls='--', lw=1, zorder=0, color='red')
            
            

def plot_pacf(ts, nlags=None, alpha=0.05, figsize=(12,4), annotate_sig=False): 
    """
    Plots the partial autocorrelation function (PACF) for a given time series.

    Parameters:
    - ts: The time series data.
    - nlags: The number of lags to include in the plot. If None, all lags will be included.
    - alpha: The significance level for determining significant lags.
    - figsize: The size of the figure (width, height) in inches.
    - annotate_sig: Whether to annotate significant lags with red dashed lines.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=figsize) 
    tsa.graphics.plot_pacf(ts, ax=ax, lags=nlags, alpha=alpha)

    if annotate_sig == True:
        sig_lags = get_sig_lags(ts,nlags=nlags,alpha=alpha, type='PACF')
        
        ## ADDING ANNOTATING SIG LAGS
        for lag in sig_lags:
            ax.axvline(lag, ls='--', lw=1, zorder=0, color='red')


def display_ts_workflow(return_text=False):
    """
    Displays the Time Series Modeling Workflow - (S)ARIMA markdown document.

    Parameters:
    - return_text (bool): If True, returns the text content of the markdown document as a string. Default is False.

    Returns:
    - str: The text content of the markdown document, if return_text is True.

    """
    import requests
    try:
        from IPython.display import Markdown, display
        display_func = lambda x: display(Markdown(x))
    except:
        display_func = print
    url = "https://raw.githubusercontent.com/coding-dojo-data-science/dojo_ds/main/assets/Time%20Series%20Modeling%20Workflow%20-%20(S)ARIMA.md"
    resp = requests.get(url)
    display_func(resp.text)

    if return_text:
        return resp.text









# def reference_interpreting_acf_pacf(seasonal=True, return_string = False):
#     table = """**Determining (S)ARIMA Orders from ACF/PACF**:\n
# |                  | AR(p)                | MA(q)               | ARMA(p,q) |
# |:----- | :---------------: | :---------------:|:---------------:|
# | **ACF**              | Gradually decreases  | Dramatic drop after lag \(q\) | Gradually decreases |
# | **PACF**             | Dramatic drop after lag \(p\) | Gradually decreases      | Gradually decreases  |
# | **ARIMA Order** (p,d,q) | (p,d,0) | (0,d,q) | Start with (1,d,1) & Iterate|

# ___

# **If seasonality is present**

# | ***If seasonal*** | S-AR(P)                | S-MA(Q)                | SARIMA(P,Q)            |
# |:-----------------| :---------------: | :---------------:|:---------------:|  
# | **ACF** (seasonal lags)†         | Gradually decreases   | Dramatic drop after lag \(Q\)| Gradually decreases | 
# | **PACF** (seasonal lags)†       |  Dramatic drop after lag \(P\) | Gradually decreases    | Gradually decreases |
# | **Seasonal Order** (P,D,Q)  | (P,D,0)      | (0,D,Q)       | Start with (1,D,1) & Iterate  | 

# - † seasonal lags = lags that are a multiple of the season length (m). E.g., If daily, m=7, check lags 7,14,21,etc.

# """
#     if return_string == True:
#         return table
#     else: 
#         from IPython.display import Markdown, display
#         display(Markdown(table))


def reference_interpreting_acf_pacf(seasonal=True, return_string=False):
    """
    Generates a table for determining (S)ARIMA orders from ACF/PACF.

    Args:
        seasonal (bool, optional): Indicates if seasonality is present. Defaults to True.
        return_string (bool, optional): Indicates if the table should be returned as a string. 
            If False, the table is displayed using IPython's Markdown. Defaults to False.

    Returns:
        str or None: The table as a string if return_string is True, otherwise None.
    """
    table = """#### **Determining (S)ARIMA Orders from ACF/PACF**:\n
|                  | AR(p)                | MA(q)               | ARMA(p,q) |
|:----- | :---------------: | :---------------:|:---------------:|
| **ACF**              | Gradually decreases  | Dramatic drop after lag \(q\) | Gradually decreases |
| **PACF**             | Dramatic drop after lag \(p\) | Gradually decreases      | Gradually decreases  |
| **ARIMA Order (p,d,q)** | **(p,d,0)** | **(0,d,q)** | **Start with (1,d,1) & Iterate**|


___


#### **Determining Seasonal Orders** (only if seasonality is present):

| ***If seasonal*** | S-AR(P)                | S-MA(Q)                | SARIMA(P,Q)            |
|:-----------------| :---------------: | :---------------:|:---------------:|  
| **ACF** (seasonal lags)†         | Gradually decreases   | Dramatic drop after lag \(Q\)| Gradually decreases | 
| **PACF** (seasonal lags)†       |  Dramatic drop after lag \(P\) | Gradually decreases    | Gradually decreases |
| **Seasonal Order (P,D,Q)**  | **(P,D,0)**      | **(0,D,Q)**       | **Start with (1,D,1) & Iterate**  | 

- † seasonal lags = lags that are a multiple of the season length (m). E.g., If daily, m=7, check lags 7,14,21,etc.

"""
    if seasonal==False:
        table = table.split('___')[0]
        table = table.replace("(S)ARIMA",'ARIMA')
   
    
             
    if return_string:
        return table
    else: 
        try:
            from IPython.display import Markdown, display
            display_func = lambda x: display(Markdown(x))
        except:
            display_func = print
            
        display_func(table)



def make_best_arima(auto_model, ts_train, exog=None, fit_kws={}):
    """
    Fits a final ARIMA model using the best parameters obtained from an auto_model and evaluates its performance.

    Parameters:
    auto_model (AutoARIMA): The AutoARIMA model object that contains the best parameters.
    ts_train (array-like): The time series data used for training the ARIMA model.
    exog (array-like, optional): Exogenous variables to be included in the model. Default is None.
    fit_kws (dict, optional): Additional keyword arguments to be passed to the `fit` method of the SARIMAX model. Default is an empty dictionary.

    Returns:
    SARIMAX: The fitted SARIMAX model with the best parameters.

    """
    best_model = tsa.SARIMAX(
        ts_train,
        exog=exog,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        sarimax_kwargs=auto_model.sarimax_kwargs,
    ).fit(disp=False, **fit_kws)
    return best_model

    
            
        
def evaluate_ts_model(model, ts_train, ts_test, exog_train=None, exog_test=None,
                      return_scores=False, show_summary=True,
                      n_train_lags=None, figsize=(9,3),
                      title='Comparing Forecast vs. True Data',
                     plot_diagnostics=True):
    """
    Evaluates a time series model by generating forecasts and comparing them to the true data.

    Parameters:
    - model: The time series model to be evaluated (Either SARIMAX or AutoARIMA object)
    - ts_train: The training time series data.
    - ts_test: The testing time series data.
    - exog_train: The exogenous variables for the training data (optional).
    - exog_test: The exogenous variables for the testing data (optional).
    - return_scores: Whether to return the model and regression metrics (optional, default=False).
    - show_summary: Whether to display the model summary and diagnostics plots (optional, default=True).
    - n_train_lags: The number of lagged values to include in the training data visualization (optional).
    - figsize: The size of the forecast plot (optional, default=(9,3)).
    - title: The title of the forecast plot (optional, default='Comparing Forecast vs. True Data').

    Returns:
    - If return_scores=True, returns the model and regression metrics.
    - If return_scores=False, returns the model.

    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ## GET FORECAST             
        # Check if auto-arima, if so, extract sarima model
        if hasattr(model, "arima_res_"):
            print(f"- Fitting a new ARIMA using the params from the auto_arima...")
            model = make_best_arima(model, ts_train, exog=exog_train)
            

                
    
        # Get forecast         
        forecast = model.get_forecast(exog=exog_test, steps=len(ts_test))
        forecast_df = forecast.summary_frame()
                        
        # Get and display the regression metrics BEFORE showing plot
        reg_res = regression_metrics_ts(ts_test, forecast_df['mean'], 
                                            output_dict=True, thiels_U=True)
        
        if show_summary==True:
            display(model.summary())
            
        if plot_diagnostics==True:
            model.plot_diagnostics(figsize=(8,4))
            plt.tight_layout()
            plt.show()
        # Visualize forecast
        plot_forecast(ts_train, ts_test, forecast_df, 
                      n_train_lags=n_train_lags, figsize=figsize,
                     title=title)
        plt.show()
    
        
    
        if return_scores:
            return model, reg_res
        else:
            return model
