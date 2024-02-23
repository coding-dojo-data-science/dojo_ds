from .evaluate import evaluate_ols, plot_residuals
def find_outliers_Z(data, verbose=True):
    """
    Find outliers based on Z-score rule (outliers have an absolute z-score that is >3)

    Args:
        data (Series): Pandas Series
        verbose (bool, optional): Print summary info about outliers. Defaults to True.

    Returns:
        Series: Boolean index for input data, where True = Outlier
    """
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    outliers = np.abs(stats.zscore(data))>3
    
    
    if verbose:
        n = len(outliers)
        print(f"- {outliers.sum():,} outliers found in {data.name} out of {n:,} rows ({outliers.sum()/n*100:.2f}%) using Z-scores.")

    outliers = pd.Series(outliers, index=data.index, name=data.name)
    return outliers


def find_outliers_IQR(data, verbose=True):
    """
    Find outliers based on IQR-rule (outliers are either 1.5 x IQR below 25% quantile and 1.5xIQR above 75% quantile).
    
    Args:
        data (Series): Pandas Series containing the data.
        verbose (bool, optional): If True, print summary information about outliers. Defaults to True.
    
    Returns:
        Series: Boolean index for input data, where True indicates an outlier.
    """
    import pandas as pd
    import numpy as np
    
    # Calculate q1 and q3 quantiles
    q3 = np.quantile(data, .75)
    q1 = np.quantile(data, .25)
    
    # Calculate IQR 
    IQR = q3 - q1
    
    # Set thresholds more than 1.5x IQR above Q3/below Q1
    upper_threshold = q3 + 1.5 * IQR
    lower_threshold = q1 - 1.5 * IQR
    
    # Identify outliers
    outliers = (data < lower_threshold) | (data > upper_threshold)
    
    if verbose:
        n = len(outliers)
        print(f"- {outliers.sum():,} outliers found in {data.name} out of {n:,} rows ({outliers.sum() / n * 100:.2f}%) using IQR.")
    
    outliers = pd.Series(outliers, index=data.index, name=data.name)
    return outliers




def remove_outliers(df_, method='iqr', subset=None, verbose=2):
    """Returns a copy of the input dataframe with outliers removed from selected columns using the specified method.

    Args:
        df_ (DataFrame): The input dataframe to copy and remove outliers from.
        method (str): The method of outlier removal. Options are 'iqr' or 'z'/'zscore'. Default is 'iqr'.
        subset (list or None): A list of column names to remove outliers from. If None, all numeric columns are used. Default is None.
        verbose (bool, int): If verbose==1, print only the overall summary. If verbose==2, print the detailed summary. Default is 2.

    Returns:
        DataFrame: A copy of the input dataframe with outliers removed.

    Raises:
        Exception: If the method is not 'iqr' or 'z'.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        >>> remove_outliers(df, method='iqr', subset=['A'], verbose=2)
        Returns a dataframe with outliers removed from column 'A' using the IQR rule.
    """
    import pandas as pd
    ## Make a copy of the input dataframe  
    df = df_.copy()
    
    ## Set verbose_func for calls to outlier funcs
    if verbose == 2:
        verbose_func = True
    else:
        verbose_func = False
        
    ## Set outlier removal function and name
    if method.lower() == 'iqr':
        find_outlier_func = find_outliers_IQR
        method_name = "IQR rule"
    elif 'z' in method.lower():
        find_outlier_func = find_outliers_Z
        method_name = 'Z-score rule'
    else:
        raise Exception('[!] Method must be either "iqr" or "z".')
        
    ## Set list of columns to remove outliers from
    if subset is None:
        col_list = df.select_dtypes('number').columns
    elif isinstance(subset, str):
        col_list = [subset]
    elif isinstance(subset, list):
        col_list = subset
    else:
        raise Exception("[!] subset must be None, a single string, or a list of strings.")

    
    ## Empty dictionary for both types of outliers
    outliers = {}

    ## Use both functions to see the comparison for the number of outliers
    for col in col_list:
        idx_outliers = find_outlier_func(df[col], verbose=verbose_func)
        outliers[col] = idx_outliers

    
    ## Getting final dataframe of all outliers to get 1 final T/F index
    outliers_combined = pd.DataFrame(outliers).any(axis=1)
    
    if verbose:
        n = len(outliers_combined)
        print(f"\n[i] Overall, {outliers_combined.sum():,} rows out of {n:,} ({outliers_combined.sum()/n*100:.2f}%) were removed as outliers using {method_name}.")
    
    
    # remove outliers 
    df_clean = df[~outliers_combined].copy()
    return df_clean
      