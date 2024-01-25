## PREVIOUS CLASSIFICATION_METRICS FUNCTION FROM INTRO TO ML

def classification_metrics(y_true, y_pred, label='',
                           output_dict=False, figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False, values_format=".2f",
                           target_names = None, return_fig=True):
    """Calculate classification metrics from preditions and display Confusion matrix.

    Args:
        y_true (Series/array): True target values.
        y_pred (Series/array): Predicted targe values.
        label (str, optional): Label For Printed Header. Defaults to ''.
        output_dict (bool, optional): Return the results of classification_report as a dict. Defaults to False.
        figsize (tuple, optional): figsize for confusion matrix subplots. Defaults to (8,4).
        normalize (str, optional): arg for sklearn's ConfusionMatrixDisplay. Defaults to 'true' (conf mat values normalized to true class).
        cmap (str, optional): Colormap for the ConfusionMatrixDisplay. Defaults to 'Blues'.
        colorbar (bool, optional): Arg for ConfusionMatrixDispaly: include colorbar or not. Defaults to False.
        values_format (str, optional): Format values on confusion matrix. Defaults to ".2f".
        target_names (array, optional): Text labels for the integer-encoded target. Passed in numeric order [label for "0", label for "1", etc.]
        return_fig (bool, optional): To get matplotlib figure for confusion matrix, set outout_dict to False and set return_fig to True.

    Returns (Only 1 value is returned):
        dict: Dictionary from classification_report. Only returned if output_dict=True.
        fig: Matplotlib figure with confusion matrix. Only returned if output_dict=False and return_fig=True
        
            
    Note: 
        This is a modified version of classification metrics function from Intro to Machine Learning.
        Updates:
          - Reversed raw counts confusion matrix cmap  (so darker==more).
          - Added arg for normalized confusion matrix values_format
    
    """
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    # Get the classification report
    report = classification_report(y_true, y_pred,target_names=target_names)
    
    ## Print header and report
    header = "-"*70
    print(header, f" Classification Metrics: {label}", header, sep='\n')
    print(report)
    
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # Create a confusion matrix  of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0], 
                                            display_labels=target_names);
    axes[0].set_title("Raw Counts")
    
    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, #New arg
                                            colorbar=colorbar,
                                            ax = axes[1],
                                            display_labels=target_names);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()
    plt.show()
    
    # Return dictionary of classification_report
    if output_dict==True:
        report_dict = classification_report(y_true, y_pred,target_names=target_names, output_dict=True)
        return report_dict

    elif return_fig == True:
        return fig
    
    
def evaluate_classification(model, X_train=None, y_train=None, X_test=None, y_test=None,
                            figsize=(6,4), normalize='true', output_dict = False,
                            cmap_train='Blues', cmap_test="Reds",colorbar=False,
                            values_format='.2f',
                            target_names=None, return_fig=False):
    """Evalutes an sklearn-compatible classification model on training and test data. 
    For each data split, return the classification report and confusion matrix display. 

    Args:
        model (sklearn estimator): Classification model to evaluate.
        X_train (Frame/Array, optional): Training data. Defaults to None.
        y_train (Series/Array, optional): Training labels. Defaults to None.
        X_test (Frame/Array, optional): Test data. Defaults to None.
        y_test (Series/Array, optional): Test labels. Defaults to None.
        figsize (tuple, optional): figsize for confusion matrix subplots. Defaults to (6,4).
        normalize (str, optional): arg for sklearn's ConfusionMatrixDisplay. Defaults to 'true' (conf mat values normalized to true class).  
        output_dict (bool, optional):  Return the results of classification_report as a dict. Defaults to False. Defaults to False.
        cmap_train (str, optional): Colormap for the ConfusionMatrixDisplay for training data. Defaults to 'Blues'.
        cmap_test (str, optional): Colormap for the ConfusionMatrixDisplay for test data.  Defaults to "Reds".
        colorbar (bool, optional): Arg for ConfusionMatrixDispaly: include colorbar or not. Defaults to False.
        values_format (str, optional): Format values on confusion matrix. Defaults to ".2f".
        target_names (array, optional): Text labels for the integer-encoded target. Passed in numeric order [label for "0", label for "1", etc.]
        return_fig (bool, optional): Whether the matplotlib figure for confusion matrix is returned. Defaults to False.
                                          Note: Must set outout_dict to False and set return_fig to True to get figure returned.


     Returns (Only 1 value is returned, but contents vary):
        dict: Dictionary that contains results for "train" and "test. 
              Contents of dictionary depending on output_dict and return_fig:
              - if output_dict==True and return_fig==False: returns dictionary of classification report results
            - if output_dict==False and return_fig==True: returns dictionary of confusion matrix displays.

    """
    # Combining arguments used for both training and test results
    shared_kwargs = dict(output_dict=output_dict,  # output_dict: Changed from hard-coded True
                      figsize=figsize, 
                      colorbar=colorbar, 
                      target_names=target_names,
                      values_format=values_format,
                      return_fig=return_fig)
 
    if (X_train is None) & (X_test is None):
        raise Exception('\nEither X_train & y_train or X_test & y_test must be provided.')
 
    if (X_train is not None) & (y_train is not None):
        # Get predictions for training data
        y_train_pred = model.predict(X_train)
        # Call the helper function to obtain regression metrics for training data
        results_train = classification_metrics(y_train, y_train_pred, cmap=cmap_train, label='Training Data', **shared_kwargs)
        print()
    else:
        results_train=None
  
    if (X_test is not None) & (y_test is not None):
        # Get predictions for test data
        y_test_pred = model.predict(X_test)
        # Call the helper function to obtain regression metrics for test data
        results_test = classification_metrics(y_test, y_test_pred,  cmap=cmap_test, label='Test Data' , **shared_kwargs)
    else:
        results_test = None
  
  
    if (output_dict == True) | (return_fig==True):
        # Store results in a dataframe if ouput_frame is True
        results_dict = {'train':results_train,
                        'test': results_test}
        return results_dict


def evaluate_classification_network(model, 
                                    X_train=None, y_train=None, 
                                    X_test=None, y_test=None,
                                    history=None, history_figsize=(6,6),
                                    figsize=(6,4), normalize='true',
                                    output_dict = False,
                                    cmap_train='Blues',
                                    cmap_test="Reds",
                                    values_format=".2f", 
                                    colorbar=False, target_names=None, 
         return_fig=False):
    """Evaluates a neural network classification task using either
    separate X and y arrays or a tensorflow Dataset

    Args:
        model (sklearn-compatible classifier): Model to evaluate.
        X_train (array or tf.data.Dataset, optional): Training data. Defaults to None.
        y_train (array, or None if X_train is a tf Dataset, optional): Training labels (if not using a tf dataset). Defaults to None.
        X_test (array or tf.data.Dataset, optional): Test data. Defaults to None.
        y_test (array, or None if X_test is a tf Dataset, optional): Test labels (if not using a tf Dataset). Defaults to None.
        history (tensorflow history object, optional): History object from model training. Defaults to None.
        history_figsize (tuple, optional): Total figure size for plot_history. Defaults to (6,8).
        figsize (tuple, optional): figsize for confusion matrix subplots. Defaults to (6,4).
        normalize (str, optional): arg for sklearn's ConfusionMatrixDisplay. Defaults to 'true' (conf mat values normalized to true class).  
        output_dict (bool, optional):  Return the results of classification_report as a dict. Defaults to False. Defaults to False.
        cmap_train (str, optional): Colormap for the ConfusionMatrixDisplay for training data. Defaults to 'Blues'.
        cmap_test (str, optional): Colormap for the ConfusionMatrixDisplay for test data.  Defaults to "Reds".
        colorbar (bool, optional): Arg for ConfusionMatrixDispaly: include colorbar or not. Defaults to False.
        values_format (str, optional): Format values on confusion matrix. Defaults to ".2f".
        target_names (array, optional): Text labels for the integer-encoded target. Passed in numeric order [label for "0", label for "1", etc.]
        return_fig (bool, optional): Whether the matplotlib figure for confusion matrix is returned. Defaults to False.
                                          Note: Must set outout_dict to False and set return_fig to True to get figure returned.


     Returns (Only 1 value is returned, but contents vary):
        dict: Dictionary that contains results for "train" and "test. 
            Contents of dictionary depending on output_dict and return_fig:
              - if output_dict==True and return_fig==False: returns dictionary of classification report results
            - if output_dict==False and return_fig==True: returns dictionary of confusion matrix displays.
    """
    if (X_train is None) & (X_test is None):
        raise Exception('\nEither X_train & y_train or X_test & y_test must be provided.')
 
    shared_kwargs = dict(output_dict=True, 
                      figsize=figsize,
                      colorbar=colorbar,
                      values_format=values_format, 
                      target_names=target_names,)
    # Plot history, if provided
    if history is not None:
        plot_history(history, figsize=history_figsize)
    ## Adding a Print Header
    print("\n"+'='*80)
    print('- Evaluating Network...')
    print('='*80)
    ## TRAINING DATA EVALUATION
    # check if X_train was provided
    if X_train is not None:
        ## Check if X_train is a dataset
        if hasattr(X_train,'map'):
            # If it IS a Datset:
            # extract y_train and y_train_pred with helper function
            y_train, y_train_pred = get_true_pred_labels(model, X_train)
        else:
            # Get predictions for training data
            y_train_pred = model.predict(X_train)
        ## Pass both y-vars through helper compatibility function
        y_train = convert_y_to_sklearn_classes(y_train)
        y_train_pred = convert_y_to_sklearn_classes(y_train_pred)
        
        # Call the helper function to obtain regression metrics for training data
        results_train = classification_metrics(y_train, y_train_pred, cmap=cmap_train,label='Training Data', **shared_kwargs)
        
        ## Run model.evaluate         
        print("\n- Evaluating Training Data:")
        print(model.evaluate(X_train, return_dict=True))
    
    # If no X_train, then save empty list for results_train
    else:
        results_train = None
  
  
    ## TEST DATA EVALUATION
    # check if X_test was provided
    if X_test is not None:
        ## Check if X_train is a dataset
        if hasattr(X_test,'map'):
            # If it IS a Datset:
            # extract y_train and y_train_pred with helper function
            y_test, y_test_pred = get_true_pred_labels(model, X_test)
        else:
            # Get predictions for training data
            y_test_pred = model.predict(X_test)
        ## Pass both y-vars through helper compatibility function
        y_test = convert_y_to_sklearn_classes(y_test)
        y_test_pred = convert_y_to_sklearn_classes(y_test_pred)
        
        # Call the helper function to obtain regression metrics for training data
        results_test = classification_metrics(y_test, y_test_pred, cmap=cmap_test,label='Test Data', **shared_kwargs)
        
        ## Run model.evaluate         
        print("\n- Evaluating Test Data:")
        print(model.evaluate(X_test, return_dict=True))
      
    # If no X_test, then save empty list for results_test
    else:
        results_test = None
      
    if (output_dict == True) | (return_fig==True):
        # Store results in a dataframe if ouput_frame is True
        results_dict = {'train':results_train,
                        'test': results_test}
        return results_dict




def plot_history(history,figsize=(6,8), return_fig=False):
    """Plots the training and validation curves for all metrics in a Tensorflow History object.

    Args:
        history (Tensorflow History): History output from training a neural network.
        figsize (tuple, optional): Total fdigure size. Defaults to (6,8).
        return_fig (boolean, optional): If true, return figure instead of displaying it with plt.show()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Get a unique list of metrics 
    all_metrics = np.unique([k.replace('val_','') for k in history.history.keys()])
    # Plot each metric
    n_plots = len(all_metrics)
    fig, axes = plt.subplots(nrows=n_plots, figsize=figsize)
    axes = axes.flatten()
    # Loop through metric names add get an index for the axes
    for i, metric in enumerate(all_metrics):
        # Get the epochs and metric values
        epochs = history.epoch
        score = history.history[metric]
        # Plot the training results
        axes[i].plot(epochs, score, label=metric, marker='.')
        # Plot val results (if they exist)
        try:
            val_score = history.history[f"val_{metric}"]
            axes[i].plot(epochs, val_score, label=f"val_{metric}",marker='.')
        except:
            pass
        finally:
            axes[i].legend()
            axes[i].set(title=metric, xlabel="Epoch",ylabel=metric)
   
    # Adjust subplots and show
    fig.tight_layout()
 
    if return_fig:
        return fig
    else:
        plt.show()

def convert_y_to_sklearn_classes(y, verbose=False):
    """Helper function to convert neural network outputs to class labels.
    if ndim ==1, use as-is.
    

    Args:
        y (array/Series): preditions to convert to classes.
        verbose (bool, optional): Print which preprocessing approach is used. Defaults to False.

    Returns:
        array: Target as 1D class labels
    """
    import numpy as np
    # If already one-dimension
    if np.ndim(y)==1:
        if verbose:
            print("- y is 1D, using it as-is.")
        return y
        
    # If 2 dimensions with more than 1 column:
    elif y.shape[1]>1:
        if verbose:
            print("- y is 2D with >1 column. Using argmax for metrics.")   
        return np.argmax(y, axis=1)
    
    else:
        if verbose:
            print("y is 2D with 1 column. Using round for metrics.")
        return np.round(y).flatten().astype(int)


def get_true_pred_labels(model,ds):
    """	Gets the labels and predicted probabilities from a Tensorflow model and Dataset object.
    Adapted from source: https://stackoverflow.com/questions/66386561/keras-classification-report-accuracy-is-different-between-model-predict-accurac
    

    Args:
        model (Tensorflow/Keras model): Model to get predictions from.
        ds (tensorflow.data.Dataset): dataset to iterate through as a numpy iterator.

    Returns:
        _type_: _description_
    """
    import numpy as np

    y_true = []
    y_pred_probs = []
    
    # Loop through the dataset as a numpy iterator
    for images, labels in ds.as_numpy_iterator():
        
        # Get prediction with batch_size=1
        y_probs = model.predict(images, batch_size=1, verbose=0)
        # Combine previous labels/preds with new labels/preds
        y_true.extend(labels)
        y_pred_probs.extend(y_probs)
    ## Convert the lists to arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    return y_true, y_pred_probs



# #### Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
def regression_metrics(y_true, y_pred, label='', verbose = True, output_dict=False):
    """Calculate MEA, MSE, RMSE, R-Squared and MAPE using the true and predicted labels.

    Args:
        y_true (Series/array): True target values.
        y_pred (Series/array): Predicted targe values.
        label (str, optional): Label to display in results header. Defaults to ''.
        verbose (bool, optional): Controls printing of results. Defaults to True. 
        output_dict (bool, optional): Return results in a dictionary. Defaults to False. (Note one of either verbose or output_dict should be set to True)

    Returns:
        dict: Dictionary of reuslts with keys: 'Label','MAE','MSE', 'RMSE', 'MAPE','R^2'. Only returned if out_dict==True.
    """
    import numpy as np
    # Get metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r_squared = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    if (verbose==False) & (output_dict==False):
     raise Exception("At least one of the following arguments must be set to True: output_dict, verbose.")

    if verbose == True:
        # Print Result with Label and Header
        header = "-"*60
        print(header, f"Regression Metrics: {label}", header, sep='\n')
        print("Relative Comparison Metrics:")
        print(f"- MAE = {mae:,.3f}")
        print(f"- MSE = {mse:,.3f}")
        print(f"- RMSE = {rmse:,.3f}")
        # print('\n')
        print("\nAbsolute Metrics")
        print(f"- MAPE = {mape:,.3f}")
        print(f"- R^2 = {r_squared:,.3f}")
  
    if output_dict == True:
        metrics = {'Label':label, 'MAE':mae,
                    'MSE':mse, 'RMSE':rmse, 'MAPE':mape,
                    'R^2':r_squared}
        return metrics



def evaluate_regression(reg, X_train, y_train, X_test, y_test, verbose = True,
                        output_frame=False):
    """Evalutes an sklearn-compatible regression model on training and test data. 
    For each data split, return the classification report and confusion matrix display. 

    Args:
        reg (sklearn estimator): Regression model to evaluate.
        X_train (Frame/Array, optional): Training data. Defaults to None.
        y_train (Series/Array, optional): Training labels. Defaults to None.
        X_test (Frame/Array, optional): Test data. Defaults to None.
        y_test (Series/Array, optional): Test labels. Defaults to None.
        verbose (bool, optional): Controls printing of results. Defaults to True. 
        output_dict (bool, optional): Return results in a dictionary. Defaults to False. (Note one of either verbose or output_dict should be set to True)

    Returns:
        dict: Dictionary of reuslts with keys: 'Label','MAE','MSE', 'RMSE', 'MAPE','R^2'. Only returned if out_dict==True.
    """
    # Get predictions for training data
    y_train_pred = reg.predict(X_train)

    # Call the helper function to obtain regression metrics for training data
    results_train = regression_metrics(y_train, y_train_pred, verbose = verbose,
                                        output_dict=output_frame,
                                        label='Training Data')
    print()
    # Get predictions for test data
    y_test_pred = reg.predict(X_test)
    # Call the helper function to obtain regression metrics for test data
    results_test = regression_metrics(y_test, y_test_pred, verbose = verbose,
                                    output_dict=output_frame,
                                        label='Test Data' )

  # Store results in a dataframe if ouput_frame is True
    if output_frame:
        import pandas as pd
        results_df = pd.DataFrame([results_train,results_test])
        # Set the label as the index
        results_df = results_df.set_index('Label')
        # Set index.name to none to get a cleaner looking result
        results_df.index.name=None
        # Return the dataframe
        return results_df.round(3)



    
def evaluate_ols(result,X_train_df, y_train, show_summary=True):
    """Plots a Q-Q Plot and residual plot for a statsmodels OLS regression, with option to display summary.

    Args:
        model (statsmodels OLS): statsmodels regression model.
        X_test_df (array/Frame): Test Data
        y_test (array/Series): Test Labels
        figsize (tuple, optional): Figsize for regression plots. Defaults to (12,5).
    """
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    try:
        from IPython.display import display
        display(result.summary())
    except:
        pass
    
    ## save residuals from result
    y_pred = result.predict(X_train_df)
    resid = y_train - y_pred
    
    fig, axes = plt.subplots(ncols=2,figsize=(12,5))
    
    ## Normality 
    sm.graphics.qqplot(resid,line='45',fit=True,ax=axes[0]);
    
    ## Homoscedasticity
    ax = axes[1]
    ax.scatter(y_pred, resid, edgecolor='white',lw=1)
    ax.axhline(0,zorder=0)
    ax.set(ylabel='Residuals',xlabel='Predicted Value');
    plt.tight_layout()
    
    
    
def plot_residuals(model,X_test_df, y_test,figsize=(12,5)):
    """Plots a Q-Q Plot and residual plot for a regression model.

    Args:
        model (regression model): regression model that supports .predict
        X_test_df (_type_): Test Data
        y_test (_type_): Test Labels
        figsize (tuple, optional): Figsize for regression plots. Defaults to (12,5).
    """

    import matplotlib.pyplot as plt
    import statsmodels.api as sm
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
    plt.tight_layout()
    
    
    
    
    