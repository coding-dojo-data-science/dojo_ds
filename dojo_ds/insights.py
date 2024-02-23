## FROM MY "From linear to logistic regression" mini-lesson (interview)
from matplotlib import ticker
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from IPython.display import display


def annotate_hbars(ax, ha='left', va='center', size=12, xytext=(4,0),
                  textcoords='offset points'):
    """
    Annotates horizontal bars on a matplotlib Axes object.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to annotate.
    - ha (str): The horizontal alignment of the annotation text. Default is 'left'.
    - va (str): The vertical alignment of the annotation text. Default is 'center'.
    - size (int): The font size of the annotation text. Default is 12.
    - xytext (tuple): The offset of the annotation text from the annotated point. Default is (4, 0).
    - textcoords (str): The coordinate system used for xytext. Default is 'offset points'.
    """
    for bar in ax.patches:
        val = bar.get_width()
        if val < 0:
            x = 0
        else:
            x = val
        bar_ax = bar.get_y() + bar.get_height()/2
        ax.annotate(f"{val:,.2f}", (x, bar_ax), ha=ha, va=va, size=size,
                    xytext=xytext, textcoords=textcoords)




from sklearn import metrics
import pandas as pd

# def evaluate_regression(model, X_train,y_train, X_test, y_test,as_frame=True):
#   """Evaluates a scikit learn regression model using r-squared and RMSE.
#   Returns the results a DataFrame if as_frame is True (Default).
#   """


#   ## Training Data
#   y_pred_train = model.predict(X_train)
#   r2_train = metrics.r2_score(y_train, y_pred_train)
#   rmse_train = metrics.mean_squared_error(y_train, y_pred_train,
#                                           squared=False)
#   mae_train = metrics.mean_absolute_error(y_train, y_pred_train)


#   ## Test Data
#   y_pred_test = model.predict(X_test)
#   r2_test = metrics.r2_score(y_test, y_pred_test)
#   rmse_test = metrics.mean_squared_error(y_test, y_pred_test,
#                                           squared=False)
#   mae_test = metrics.mean_absolute_error(y_test, y_pred_test)

#   if as_frame:
#       df_version =[['Split','R^2','MAE','RMSE']]
#       df_version.append(['Train',r2_train, mae_train, rmse_train])
#       df_version.append(['Test',r2_test, mae_test, rmse_test])
#       df_results = pd.DataFrame(df_version[1:], columns=df_version[0])
#       df_results = df_results.round(2)


#       # adapting hide_index for pd version
#       if pd.__version__ < "1.4.0":
#         display(df_results.style.hide_index().format(precision=2, thousands=','))
#       else:
#         display(df_results.style.hide(axis='index').format(precision=2, thousands=','))

#   else:
#       print(f"Training Data:\tR^2 = {r2_train:,.2f}\tRMSE = {rmse_train:,.2f}\tMAE = {mae_train:,.2f}")
#       print(f"Test Data:\tR^2 = {r2_test:,.2f}\tRMSE = {rmse_test:,.2f}\tMAE = {mae_test:,.2f}")



def get_coefficients(reg, name='Coefficients'):
    """Save a model's .coef_ and .intercept_ as a Pandas Series"""
    raise Exception("Deprecated - use get_coeffs_linreg instead")

    # coeffs = pd.Series(reg.coef_,
    #                    index= reg.feature_names_in_,
    #                    name=name)

    # if reg.intercept_ != 0.0:
    #     coeffs.loc['Intercept'] = reg.intercept_
    # return coeffs
    
def get_coeffs_linreg(lin_reg, feature_names=None, sort=True, ascending=True,
                      name='LinearRegression Coefficients'):
    """
    Get the coefficients of a linear regression model.

    Parameters:
    - lin_reg: The trained linear regression model.
    - feature_names: Optional. The names of the features used in the model. If not provided, it will use the feature names from the model.
    - sort: Optional. Whether to sort the coefficients by value. Default is True.
    - ascending: Optional. Whether to sort the coefficients in ascending order. Default is True.
    - name: Optional. The name of the coefficients series. Default is 'LinearRegression Coefficients'.

    Returns:
    - coeffs: A pandas Series containing the coefficients of the linear regression model.
    """
    if feature_names is None:
        feature_names = lin_reg.feature_names_in_
    ## Saving the coefficients
    coeffs = pd.Series(lin_reg.coef_, index=feature_names)
    coeffs['intercept'] = lin_reg.intercept_
    if sort == True:
        coeffs = coeffs.sort_values(ascending=ascending)
    return coeffs


def plot_coefficients(coeffs, figsize=(6,5), title='Regression Coefficients',
                      intercept=True, intercept_name='Intercept',
                      sort_values=True, ascending=True,
                      ):
    raise Exception("Deprecated: use plot_coeffs instead.")

#     ## Exclude intercept if intercept==False
#     if intercept==False:
#         if intercept_name in coeffs:
#             coeffs = coeffs.drop(intercept_name).copy()

#     ## Sort values
#     if sort_values:
#         ceoffs = coeffs.sort_values(ascending=ascending)

#     ## Plot
#     ax = ceoffs.plot(kind='barh',figsize=figsize)

#     ## Customize Viz
#     ax.axvline(0,color='k', lw=1)
#     ax.set(ylabel='Feature Name',xlabel='Coefficient',title=title)
#     return ax

def plot_coeffs(coeffs, top_n=None, figsize=(4,5), 
                intercept=False, intercept_name="intercept", 
                annotate=False, ha='left', va='center', size=12, 
                xytext=(4,0), textcoords='offset points'):
    """ Plots the top_n coefficients from a Series, with optional annotations.
    
    Parameters:
    coeffs (pd.Series): The coefficients to be plotted.
    top_n (int, optional): The number of top coefficients to plot. If None, all coefficients will be plotted. Default is None.
    figsize (tuple, optional): The size of the figure. Default is (4, 5).
    intercept (bool, optional): Whether to include the intercept coefficient in the plot. Default is False.
    intercept_name (str, optional): The name of the intercept coefficient. Default is "intercept".
    annotate (bool, optional): Whether to annotate the coefficients on the plot. Default is False.
    ha (str, optional): The horizontal alignment of the annotations. Default is 'left'.
    va (str, optional): The vertical alignment of the annotations. Default is 'center'.
    size (int, optional): The font size of the annotations. Default is 12.
    xytext (tuple, optional): The offset of the annotations from the data points. Default is (4, 0).
    textcoords (str, optional): The coordinate system used for the annotations. Default is 'offset points'.
    
    Returns:
    matplotlib.axes.Axes: The plot of the coefficients.
    """
    # Drop intercept if intercept=False and 
    if (intercept == False) & (intercept_name in coeffs.index):
        coeffs = coeffs.drop(intercept_name)
    if top_n == None:
        ## sort all features and set title
        plot_vals = coeffs.sort_values()
        title = "All Coefficients - Ranked by Magnitude"
    else:
        ## rank the coeffs and select the top_n
        coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
        top_n_features = coeff_rank.head(top_n)
        
        ## sort features and keep top_n and set title
        plot_vals = coeffs.loc[top_n_features.index].sort_values()
        title = f"Top {top_n} Largest Coefficients"
    ## plotting top N importances
    ax = plot_vals.plot(kind='barh', figsize=figsize)
    ax.set(xlabel='Coefficient', 
            ylabel='Feature Names', 
            title=title)
    ax.axvline(0, color='k')
    if annotate == True:
        annotate_hbars(ax, ha=ha, va=va, size=size, xytext=xytext, textcoords=textcoords)
    return ax




def plot_residuals(model,X_test_df, y_test,figsize=(12,5)):
    """
    Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.

    Parameters:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted regression model.
    X_test_df (pandas.DataFrame): The test dataset features.
    y_test (array-like): The test dataset target variable.
    figsize (tuple, optional): The size of the figure. Defaults to (12,5).

    Returns:
    None
    """
    ## Make predictions and calculate residuals
    y_pred = model.predict(X_test_df)
    resid = y_test - y_pred

    fig, axes = plt.subplots(ncols=2,figsize=figsize)

    ## Normality
    sm.graphics.qqplot(resid, line='45',fit=True,ax=axes[0]);

    ## Homoscedascity
    ax = axes[1]
    ax.scatter(y_pred, resid, edgecolor='white',lw=0.5)
    ax.axhline(0,zorder=0)
    ax.set(ylabel='Residuals',xlabel='Predicted Value');
    fig.tight_layout()
    plt.show()



def summarize_df(df_):
    """
    Summarizes a DataFrame by providing insights on column data types, null values, unique values, and numeric range.

    Parameters:
    df_ (pandas.DataFrame): The DataFrame to be summarized.

    Returns:
    pandas.DataFrame: A summary report DataFrame with the following columns:
        - 'Column': The column names of the DataFrame.
        - 'dtype': The data types of the columns.
        - '# null': The number of null values in each column.
        - 'null (%)': The percentage of null values in each column.
        - 'nunique': The number of unique values in each column.
        - 'min': The minimum numeric value in each column.
        - 'max': The maximum numeric value in each column.

    Example Usage:
    >> df = pd.read_csv(filename)
    >> summary = summarize_df(df)
    """
    df = df_.copy()
    report = pd.DataFrame({
                        'dtype':df.dtypes,
                        '# null': df.isna().sum(),
                        'null (%)': df.isna().sum()/len(df)*100,
                        'nunique':df.nunique(),
                        "min":df.min(numeric_only=True),
                        'max':df.max(numeric_only=True)
             })
    report.index.name='Column'

    with pd.option_context("display.min_rows", len(df)):
        display(report.round(2))
    return report.reset_index()


def get_importances(model, feature_names=None, name='Feature Importance',
                   sort=False, ascending=True):
    """
    Extract the feature importances for a given model.

    Parameters:
    model (object): The trained model for which feature importances are calculated.
    feature_names (list, optional): List of feature names. If not provided, it will be extracted from the model.
    name (str, optional): Name of the feature importances series. Default is 'Feature Importance'.
    sort (bool, optional): Whether to sort the importances in ascending order. Default is False.
    ascending (bool, optional): Whether to sort the importances in ascending order. Default is True.

    Returns:
    importances (pd.Series): Series containing the feature importances.
    """
    ## checking for feature names
    if feature_names == None:
        feature_names = model.feature_names_in_

    ## Saving the feature importances
    importances = pd.Series(model.feature_importances_, index= feature_names,
                           name=name)

    # sort importances
    if sort == True:
        importances = importances.sort_values(ascending=ascending)

    return importances




def plot_importance(importances, top_n=None,  figsize=(8,6)):
    """
    Plots the importance of features in a horizontal bar chart.

    Parameters:
    importances (pandas.Series): The importance values of the features.
    top_n (int, optional): The number of top most important features to plot. If None, all features will be plotted. Default is None.
    figsize (tuple, optional): The size of the figure. Default is (8, 6).

    Returns:
    matplotlib.axes.Axes: The axes object of the plot.

    """
    # sorting with asc=false for correct order of bars
    if top_n==None:
        ## sort all features and set title
        plot_vals = importances.sort_values()
        title = "All Features - Ranked by Importance"
    else:
        ## sort features and keep top_n and set title
        plot_vals = importances.sort_values().tail(top_n)
        title = f"Top {top_n} Most Important Features"
    ## plotting top N importances
    ax = plot_vals.plot(kind='barh', figsize=figsize)
    ax.set(xlabel='Importance',
            ylabel='Feature Names',
            title=title)
    ## return ax in case want to continue to update/modify figure
    return ax



def get_color_dict(importances, color_rest='#006ba4' , color_top='green',
                    top_n=7):
    """
    Returns a dictionary mapping feature names to colors based on their importances.

    Parameters:
    importances (pd.Series): A pandas Series containing feature importances.
    color_rest (str, optional): The color code for non-highlighted features. Defaults to '#006ba4'.
    color_top (str, optional): The color code for highlighted features. Defaults to 'green'.
    top_n (int, optional): The number of top features to highlight. Defaults to 7.

    Returns:
    dict: A dictionary mapping feature names to colors.
    """
    highlight_feats = importances.sort_values(ascending=True).tail(top_n).index
    colors_dict = {col: color_top if col in highlight_feats else color_rest for col in importances.index}
    return colors_dict


def plot_importance_color(importances, top_n=None,  figsize=(8,6), 
                          color_dict=None, ax=None):
    """
    Plot the feature importances with optional color highlighting.

    Parameters:
    - importances (pandas.Series): The feature importances.
    - top_n (int, optional): The number of top features to display. If None, all features will be displayed. Default is None.
    - figsize (tuple, optional): The figure size. Default is (8, 6).
    - color_dict (dict, optional): A dictionary mapping feature names to colors for highlighting. Default is None.
    - ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes will be created. Default is None.

    Returns:
    - ax (matplotlib.axes.Axes): The axes object containing the plot.

    Example Use:
    fig, axes = plt.subplots(ncols=2, figsize=(20,8))
    n = 20 # setting the # of features to use for both subplots
    
    plot_importance_color(importances, top_n=n, ax=axes[0], color_dict= colors_top7)
    axes[0].set(title='R.F. Importances')

    plot_importance_color(permutation_importances, top_n=n, ax=axes[1], color_dict=colors_top7)
    axes[1].set(title='Permutation Importances')
    fig.tight_layout()
    """
    # sorting with asc=false for correct order of bars
    if top_n==None:
        ## sort all features and set title
        plot_vals = importances.sort_values()
        title = "All Features - Ranked by Importance"
    else:
        ## sort features and keep top_n and set title
        plot_vals = importances.sort_values().tail(top_n)
        title = f"Top {top_n} Most Important Features"
    ## create plot with colors, if provided
    if color_dict is not None:
        ## Getting color list and saving to plot_kws
        colors = plot_vals.index.map(color_dict)
        ax = plot_vals.plot(kind='barh', figsize=figsize, color=colors, ax=ax)
        
    else:
        ## create plot without colors, if not provided
        ax = plot_vals.plot(kind='barh', figsize=figsize, ax=ax)
        
    # set titles and axis labels
    ax.set(xlabel='Importance', 
           ylabel='Feature Names', 
           title=title)
    
    ## return ax in case want to continue to update/modify figure
    return ax


# def get_coeffs_logreg(logreg, feature_names = None, sort=True,ascending=True,
#                       name='LogReg Coefficients', class_index=0):
#     if feature_names is None:
#         feature_names = logreg.feature_names_in_ 
    
#     ## Saving the coefficients
#     coeffs = pd.Series(logreg.coef_[class_index],
#                        index= feature_names, name=name)
    
#     # use .loc to add the intercept to the series
#     coeffs.loc['intercept'] = logreg.intercept_[class_index]
#     if sort == True:
#         coeffs = coeffs.sort_values(ascending=ascending)  
#     return coeffs



def get_coeffs_logreg(logreg, feature_names = None, sort=True,ascending=True,
                      name='LogReg Coefficients', class_index=0,  
                      include_intercept=True, as_odds=False):
    """
    Get the coefficients of a logistic regression model.

    Parameters:
    logreg (object): The logistic regression model.
    feature_names (list, optional): List of feature names. If None, it uses the feature names from the model.
    sort (bool, optional): Whether to sort the coefficients. Default is True.
    ascending (bool, optional): Whether to sort the coefficients in ascending order. Default is True.
    name (str, optional): Name of the coefficients. Default is 'LogReg Coefficients'.
    class_index (int, optional): Index of the class for which to get the coefficients. Default is 0.
    include_intercept (bool, optional): Whether to include the intercept in the coefficients. Default is True.
    as_odds (bool, optional): Whether to exponentiate the coefficients to obtain odds ratios. Default is False.

    Returns:
    pd.Series: Series containing the coefficients.
    """
    
    if feature_names is None:
        feature_names = logreg.feature_names_in_
        
    ## Saving the coefficients
    coeffs = pd.Series(logreg.coef_[class_index],
                       index= feature_names, name=name)
    
    if include_intercept:
        # use .loc to add the intercept to the series
        coeffs.loc['intercept'] = logreg.intercept_[class_index]
        
    if as_odds==True:
        coeffs = np.exp(coeffs)
    if sort == True:
        coeffs = coeffs.sort_values(ascending=ascending)
    
        
    return coeffs





def plot_coeffs_color(coeffs, top_n=None,  figsize=(8,6), legend_loc='best',
                      threshold=None, color_lt='darkred', color_gt='forestgreen',
                      color_else='gray', label_thresh='Equally Likely',
                      label_gt='More Likely', label_lt='Less Likely',
                   plot_kws = {}):
    """Plots series of coefficients
    
    Args:
        coeffs (pandas Series): Importance values to plot.
        top_n (int): The number of features to display (Default=None).
                     If None, display all. Otherwise, display top_n most important.
        figsize (tuple): figsize tuple for .plot.
        legend_loc (str): Location of the legend in the plot (Default='best').
        threshold (float): Threshold value for coloring the coefficients (Default=None).
        color_lt (str): Color for coefficients less than the threshold (Default='darkred').
        color_gt (str): Color for coefficients greater than the threshold (Default='forestgreen').
        color_else (str): Color for coefficients that do not meet the threshold (Default='gray').
        label_thresh (str): Label for the threshold line in the legend (Default='Equally Likely').
        label_gt (str): Label for coefficients greater than the threshold in the legend (Default='More Likely').
        label_lt (str): Label for coefficients less than the threshold in the legend (Default='Less Likely').
        plot_kws (dict): Additional keyword arguments accepted by pandas' .plot method.
        
    Returns:
        matplotlib.axes._subplots.AxesSubplot: Matplotlib axis object.
    """
    # sorting with asc=false for correct order of bars
    if top_n is None:
        ## sort all features and set title
        plot_vals = coeffs.sort_values()
        title = "All Coefficients"
    else:
        ## rank the coeffs and select the top_n
        coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
        top_n_features = coeff_rank.head(top_n)
        plot_vals = coeffs.loc[top_n_features.index].sort_values()
        ## sort features and keep top_n and set title
        title = f"Top {top_n} Largest Coefficients"
        ## plotting top N importances
    if threshold is not None:
        color_dict = get_colors_gt_lt(plot_vals, threshold=threshold,
                                      color_gt=color_gt,color_lt=color_lt,
                                      color_else=color_else)
        ## Getting color list and saving to plot_kws
        colors = plot_vals.index.map(color_dict)
        plot_kws.update({'color':colors})
    
    
    ax = plot_vals.plot(kind='barh', figsize=figsize,**plot_kws)
    ax.set(xlabel='Coefficient',
            ylabel='Feature Names',
            title=title)
    
    if threshold is not None:
        ln1 = ax.axvline(threshold,ls=':',color='black')
        from matplotlib.patches import Patch
        box_lt = Patch(color=color_lt)
        box_gt = Patch(color=color_gt)
        handles = [ln1,box_gt,box_lt]
        labels = [label_thresh,label_gt,label_lt]
        ax.legend(handles,labels, loc=legend_loc)
    ## return ax in case want to continue to update/modify figure
    return ax



def get_colors_gt_lt(coeffs, threshold=1, color_lt ='darkred',
                     color_gt='forestgreen', color_else='gray'):
    """
    Creates a dictionary of features and their corresponding colors based on whether the value is greater than or less than the threshold.

    Parameters:
    coeffs (pandas.DataFrame): The coefficients dataframe.
    threshold (float): The threshold value. Default is 1.
    color_lt (str): The color for values less than the threshold. Default is 'darkred'.
    color_gt (str): The color for values greater than the threshold. Default is 'forestgreen'.
    color_else (str): The color for values equal to the threshold. Default is 'gray'.

    Returns:
    dict: A dictionary mapping features to their respective colors.
    """
    colors_dict = {}
    for i in coeffs.index:
        rounded_coeff = np.round( coeffs.loc[i],3)
        if rounded_coeff < threshold:
            color = color_lt
        elif rounded_coeff > threshold:
            color = color_gt
        else:
            color=color_else
        colors_dict[i] = color
    return colors_dict


