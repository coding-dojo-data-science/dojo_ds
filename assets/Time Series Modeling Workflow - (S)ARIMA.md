# Time Series Modeling Workflow - (S)ARIMA


## 1. Import libraries and Custom Functions

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import pmdarima as pm
from pmdarima.arima.utils import ndiffs, nsdiffs, diff, diff_inv
from pmdarima.model_selection import train_test_split
import pmdarima as pm
plt.rcParams['figure.figsize']=(12,3)
```

We have created several custom functions that must be defined in your notebook before calling.
- plot_forecast
- regression_metrics_ts
- get_adfuller_results
- plot_acf_pacf

## 2. Load and Explore Data

Load your time series data into a Pandas DataFrame and explore its structure, summary statistics, and any trends or patterns. Prepare the date time index and set a frequency.

```python
## Load the data
fname =''
df = pd.read_csv(fname,
                 # use args below if know datetime column already
#                 parse_dates=[''], index_col="" 
                )
df
```

- Make sure the data has a datetime index (note: this process may be quick or involved depending on the format of your starting data).

```
# make sure index is datetime index

```

**Define Time Series for Modeling**

- Decide how much data to include.
    - How far into the future do you want to forecast? (Your test split needs to be at least this long)

- Decide the final timespan to use for modeling (more isn't always better) and save as `ts`.

```python
# Select time series to model.
col = '' # if a dataframe
ts = df[col]
ts
```

**Set Frequency**

What frequency is your data in? Is this the frequency you want to use for modeling?

```python
# check frequency
ts.index
```

If needed, resample to the desired frequency

```python
# resample to desired frequency
# ts = ts.resample(...)..agg()
# ts
```

**Visualize Time Series**

```python
# Visualize selected time series
ax = ts.plot()
```

## 3. Handle Missing Values

Check for missing values in your time series data. If any are present, decide on an appropriate method for handling them, such as interpolation or forward/backward filling. - These approaches are acceptable to perform before a train test split.

```python
# Check for null values
ts.isna().sum()
## Impute null values (if any)
# ts = ts.fillna(0)
# ts = ts.interpolate()
# ts = ts.fillna(method='ffill')
# ts = ts.fillna(method='bfill')
```

## 4. Determine if a seasonal or non-seasonal model is appropriate for the data

If you suspect seasonality, decompose your time series into its constituent components—trend, seasonality, and residuals—using techniques like seasonal decomposition.

This step helps identify the underlying patterns, such as:

- Is there significant seasonality to the data?
    - Are the heights of the seasonal component large or small compared to the original time series units?
- If so, what is the length of a season (m)?
    - Seasons are usually based on the frequency of the time series. For example, if the data is:
        - daily: m = 365
            - `could season be a month (m=30)?`
        - weekly: m = 52
            - `could season be a quarter? (m=12)`
        - monthly: m = 12

```python
## Use Seasonal Decompose to check for seasonality
decomp = tsa.seasonal_decompose(ts)
fig = decomp.plot()
fig.set_size_inches(9, 5)
fig.tight_layout()
```


#### Determine the magnitude of the seasonal component relative to the range of data

```python
# How big is the seasonal component
seasonal_delta = decomp.seasonal.max() - decomp.seasonal.min()

# How big is the seasonal component relative to the time series?
print(f"The seasonal component is {seasonal_delta} which is ~{seasonal_delta/(ts.max()-ts.min()) * 100 :.2f}% of the variation in time series.")
```

#### Determine the length of a season (m)

```python
# zooming in on smaller time period to see length of season
# decomp.seasonal.loc["...":].plot();
```



## 5. Check Stationarity and determine differencing (d and D)

Assess the stationarity of your time series. Stationarity is a critical assumption for many time series models. You can use statistical tests like the Augmented Dickey-Fuller (ADF) test to check for stationarity.

- If the data is not stationary, you might need to apply transformations such as differencing and/or seasonal differencing to achieve stationarity.

**Stationarity Observations:**

- Is the raw data stationary?

- If not, does applying differencing (.diff()) make it stationary?:

    - Apply single-order differencing and test again (`ts_diff = ts.diff().dropna()`)
    - If still not stationary, apply second-order differencing and try again (`ts_diff2 = ts.diff().diff().dropna()`
    - For seasonal differencing, include m such as: ts_diff = ts.diff(m).dropna()

- If applying differencing achieved stationarity:

    - Take the order of the differencing (1 or 2) as `d`
    - Use the differenced time series for following EDA steps (ACF/PACF)

```python
# Check for stationarity
get_adfuller_results(ts)
```

- Or Programmatically determine d and D

```python
# Determine differencing
d = ndiffs(ts)
print(f'd is {d}')
D = nsdiffs(ts, m = _)
print(f'D is {D}')
```

- Apply the differencing

```python
# For example, one non seasonal differencing

# If using pandas
# ts_diff = ts.diff().dropna()

# If using pmdarima's diff
# ts_diff = diff(ts, differences=d)

```

## 6. Check Autocorrelation and Partial Autocorrelation to determine initial orders

Examine the autocorrelation function (ACF) and partial autocorrelation function (PACF) plots to understand the correlation between lagged observations. These plots can help you identify the order of autoregressive (AR), moving average (MA), and seasonal components in your time series.

## Estimate Initial Orders with ACF and PACF


![img](https://github.com/coding-dojo-data-science/dojo_ds/blob/main/assets/acf-pacf-table.png?raw=true)

Use the differenced data for your plots.

When determining the seasonal orders, only consider the seasonal lags.

Use the custom function and annotate seasons if applicable

```python
plot_acf_pacf(ts_diff, annotate_seas=True, m = _);
```

## 7. Split into Training and Test Sets

Divide your data into a training set and a separate test set. The training set is used to fit the model, while the test set is used for evaluating the model's performance.

**Use the original, non-differenced time series**, since the ts will be differenced as part of modeling.

Test size can be either:

- a percentage of the data (test_size = .20)
- a particular number of lags (test_size = 6)

We recommend visualizing the train test split.

```python
from pmdarima.model_selection import train_test_split
train, test = train_test_split(ts, test_size=___)

## Visualize train-test-split
ax = train.plot(label="train")
test.plot(label="test")
ax.legend();
```
## 8. Define the Time Series Model Orders and Fit the model to the training data

Select an appropriate time series model based on the characteristics of your data, the results of stationarity and autocorrelation analyses (make ACF/PACF plots), and the identified patterns. Models include Autoregressive Integrated Moving Average (ARIMA), Seasonal ARIMA (SARIMA), etc.

```python
# Orders for non seasonal components
p = _  # nonseasonal AR
d = _  # nonseasonal differencing
q = _ # nonseasonal MA

# Orders for seasonal components (if seasonal model)
P = _  # Seasonal AR
D = _  # Seasonal differencing
Q = _  # Seasonal MA
m = _ # Seasonal period

sarima = tsa.ARIMA(train, order = (p,d,q), seasonal_order=(P,D,Q,m)).fit()
```

## Check the model summary. 

```python
# Obtain summary
sarima.summary()
```

Check the diagnostic plots.

```python
# Obtain diagnostic plots
fig = sarima.plot_diagnostics()
fig.set_size_inches(10,6)
fig.tight_layout()
```

## 9. Generate and Visualize Forecasts 

Once you are satisfied with the model's performance, utilize it to generate forecasts for future time periods. - Specify the desired forecast horizon and obtain point estimates for the future values. - Note: Make sure not to forecast any farther into the future than the number of time lags included in the test data.

Use the custom plot_forecast function.

```python
# Obtain summary of forecast as dataframe
forecast_df = sarima.get_forecast(len(test)).summary_frame()
# Plot the forecast with true values
plot_forecast(train, test, forecast_df, n_train_lags = 50)
```


**10. Evaluate Model Performance**

Assess the performance of your fitted model by comparing its predictions to the actual values in the test set. Use appropriate evaluation metrics such as mean squared error (MSE), mean absolute error (MAE), root mean squared error (RMSE), or R-squared.

```python
regression_metrics_ts(test, forecast_df["forecast"])
```

## 11. Iterate as Needed 

Remember that selecting the appropriate model depends on the characteristics of your specific time series data, and it may require iterations and adjustments to achieve the best results. There are several options for trying alternative models:

- Manually fit other models
- Loop through variations of the model
- Use pmdarima's auto_arima

## 12. Grid Search Orders with pmdarima 

When using auto_arima, you must know if your model is nonseasonal or seasonal (and define the period, m).

```python
import pmdarima as pm
# Default auto_arima will select model based on AIC score
auto_model = pm.auto_arima(
    train,
    seasonal=___,  # True or False
    m=____,  # if seasonal
    trace=True
)
```



## 13. Fit Statsmodels SARIMA Model Using the Parameters from auto_arima

Make a new statsmodels SARIMA model using the auto_arima's `auto_arima.order`and `auto_arima.seasonal_order` to set parameters for the new model.

- Fit the model on the training data and fully evaluate using the test data & forecast.

```python
# Try auto_arima orders
sarima = tsa.ARIMA(train, order = auto_model.order, seasonal_order=auto_model.seasonal_order).fit()

# Obtain summary
sarima.summary()
```

Obtain model diagnostics and forecasts as before.

##  

## 14. Select and justify the final model

Consider the summary, diagnostic plots, AIC, BIC, and regression metrics. 

##  

## 15. Fit a final model on the entire time series

Once you have selected the best model parameters/orders, train one final model iteration of the best model using the entire time series (train and test combined).

```python
final_p = "?"
final_q = "?"
final_d = "?"
final_P = "?"
final_Q = "?"
final_D = "?"

final_model = tsa.ARIMA(
    ts,
    order=(final_p, final_d, final_q),
    seasonal_order=(final_P, final_D, final_Q, m),
).fit()
```



## 16. Generate and Visualize Future Forecasts

Since the final model was trained on both training and test data, it can now generate forecasts into the future, beyond the training data. - Rule of thumb: never forecast farther into the future than the length of the original test data. Finally, visualize the observed values, fitted values, and forecasted values, along with the confidence intervals. This step helps communicate the results and provides insights into the uncertainty of the forecasts.

```python
# Ger forecast into true future (fit on entrie time series)
forecast_df = final_model.get_forecast(len(test)).summary_frame()

plot_forecast(train, test, forecast_df, n_train_lags = 20);
```


## 17. Calculate Summary Metrics for Stakeholder (Optional)

This will vary depending on your stakeholder needs, but here are a few examples.

```python
# Define starting and final values
starting_value = forecast_df['mean'].iloc[0]
final_value = forecast_df['mean'].iloc[-1]
# Change in x
delta = final_value - starting_value
print(f'The change in X over the forecast is {delta: .2f}.')
perc_change = (delta/starting_value) *100
print (f'The percentage change is {perc_change :.2f}%.')
```

## Summary

Remember that selecting the appropriate model depends on the characteristics of your specific time series data, and it may require iterations and adjustments to achieve the best results.