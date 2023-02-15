"""Functions directly from LP stack 5 week 1: Model insights"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
############################################################################################
############################### FROM: INSIGHTS FOR STAKEHODLERS ############################
############################################################################################

def summarize_df(df_):
    """Source: Insights for Stakeholder Lesson - https://login.codingdojo.com/m/0/13079/91969 
    Example Usage:
    >> df = pd.read_csv(filename)
    >> summary = summarize_df(df)
    >> summary"""
    df = df_.copy()
    report = pd.DataFrame({
                        'dtype':df.dtypes,
                        '# null': df.isna().sum(),
                        'null (%)': df.isna().sum()/len(df)*100,
                        'nunique':df.nunique(),
                        "min":df.min(),
                        'max':df.max()
             })
    report.index.name='Column'
    return report.reset_index()

############################################################################################
############################### FROM: FEATURE IMPORTANCE ###################################
############################################################################################

def evaluate_regression(model, X_train,y_train, X_test, y_test): 
    """Evaluates a scikit learn regression model using r-squared and RMSE
    Source: Feature Importance Lesson: https://login.codingdojo.com/m/0/13079/97711
    
    Example Usage:
    >> reg = RandomForestRegressor()
    >> reg.fit(X_train_df,y_train)
    >> evaluate_regression(reg, X_train_df, y_train, X_test_df, y_test)
    """
    
    ## Training Data
    y_pred_train = model.predict(X_train)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    rmse_train = metrics.mean_squared_error(y_train, y_pred_train, 
                                            squared=False)
    
    print(f"Training Data:\tR^2= {r2_train:.2f}\tRMSE= {rmse_train:.2f}")
        
    
    ## Test Data
    y_pred_test = model.predict(X_test)
    r2_test = metrics.r2_score(y_test, y_pred_test)
    rmse_test = metrics.mean_squared_error(y_test, y_pred_test, 
                                            squared=False)
    
    print(f"Test Data:\tR^2= {r2_test:.2f}\tRMSE= {rmse_test:.2f}")




def get_importances(model, feature_names=None,name='Feature Importance',
                   sort=False, ascending=True):
    """Source: Feature Importance Lesson: https://login.codingdojo.com/m/0/13079/97711
    Example Use:
    >> reg = RandomForestRegressor()
    >> reg.fit(X_train,y_train)
    >> importances = get_importances(reg,sort=True,ascending=False)
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
    """Source: Feature Importance Lesson: https://login.codingdojo.com/m/0/13079/97711
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


############################################################################################
############################### FROM: PERMUTATION IMPORTANCE ###############################
############################################################################################

def get_color_dict(importances, color_rest='#006ba4' , color_top='green',
                    top_n=7):
    """Source: Permutation Importance Lesson - https://login.codingdojo.com/m/0/13079/101057
    
    Example Use:
    >> importances - get_importances(rf_reg)
    >> colors_top7 = get_color_dict(importances, top_n=7)
    >> colors = importances.index.map(color_dict)
    >> ax = importances.plot(kind='barh', color=colors)
    """
    ## color -coding top 5 bars
    highlight_feats = importances.sort_values(ascending=True).tail(top_n).index
    colors_dict = {col: color_top if col in highlight_feats else color_rest for col in importances.index}
    return colors_dict


def plot_importance_color(importances, top_n=None,  figsize=(8,6), 
                          color_dict=None):
    """Source: Permutation Importance Lesson - https://login.codingdojo.com/m/0/13079/101057
    
    Example Use:
    >> importances - get_importances(rf_reg)
    >> colors_top7 = get_color_dict(importances, top_n=7)
    >> ax = plot_importance_color(permutation_importances,color_dict=colors_top7,top_n=20);
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
        ax = plot_vals.plot(kind='barh', figsize=figsize, color=colors)
        
    else:
        ## create plot without colors, if not provided
        ax = plot_vals.plot(kind='barh', figsize=figsize)
        
    # set titles and axis labels
    ax.set(xlabel='Importance', 
           ylabel='Feature Names', 
           title=title)
    
    ## return ax in case want to continue to update/modify figure
    return ax




def plot_importance_color_ax(importances, top_n=None,  figsize=(8,6), 
                          color_dict=None, ax=None):
    """Source: Permutation Importance Lesson - https://login.codingdojo.com/m/0/13079/101057
    
    Example Use:
    >> importances - get_importances(rf_reg)
    >> colors_top7 = get_color_dict(importances, top_n=7)
    >> fig, axes = plt.subplots(ncols=2, figsize=(20,8)) 

    >> plot_importance_color_ax(importances, top_n=20, ax=axes[0],
                                 color_dict= colors_top7)
                                 
    >> plot_importance_color_ax(permutation_importances, top_n=20, ax=axes[1],
                                 color_dict=colors_top7)
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


############################################################################################
################### FROM: Linear Regression Coefficients - Revisited  ######################
############################################################################################

def get_coeffs_linreg(lin_reg, feature_names = None, sort=True,ascending=True,
                     name='LinearRegression Coefficients'):
    """Source: https://login.codingdojo.com/m/0/13079/99064
    """
    if feature_names is None:
        feature_names = lin_reg.feature_names_in_
        
    ## Saving the coefficients
    coeffs = pd.Series(lin_reg.coef_, index= feature_names)
    coeffs['intercept'] = lin_reg.intercept_
    if sort==True:
        coeffs = coeffs.sort_values(ascending=ascending)
        
    return coeffs



############################################################################################
############################# FROM: Visualizing Coefficients ###############################
############################################################################################


# def plot_coeffs(coeffs, top_n=None,  figsize=(4,5), intercept=False):
#     """Source: https://login.codingdojo.com/m/0/13079/101234"""
#     if (intercept==False) & ('intercept' in coeffs.index):
#         coeffs = coeffs.drop('intercept')
        
#     if top_n==None:
#         ## sort all features and set title
#         plot_vals = coeffs#.sort_values()
#         title = "All Coefficients - Ranked by Magnitude"
#     else:
#         ## rank the coeffs and select the top_n
#         coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
#         top_n_features = coeff_rank.head(top_n)
#         plot_vals = coeffs.loc[top_n_features.index].sort_values()
#         ## sort features and keep top_n and set title
#         title = f"Top {top_n} Largest Coefficients"
        
#     ## plotting top N importances
#     ax = plot_vals.plot(kind='barh', figsize=figsize)
#     ax.set(xlabel='Coefficient', 
#            ylabel='Feature Names', 
#            title=title)
#     ax.axvline(0, color='k')
    
#     ## return ax in case want to continue to update/modify figure
#     return ax


def annotate_hbars(ax, ha='left',va='center',size=12,  xytext=(4,0),
                  textcoords='offset points'):
    """Source: https://login.codingdojo.com/m/0/13079/101234
    Example Use:
    >> ax = plot_coeffs(coeffs, top_n=15)
    >> annotate_hbars(ax)
    """
    for bar in ax.patches:
    
        ## calculate center of bar
        bar_ax = bar.get_y() + bar.get_height()/2
        ## get the value to annotate
        val = bar.get_width()
        if val < 0:
            val_pos = 0
        else:
            val_pos = val
        # ha and va stand for the horizontal and vertical alignment
        ax.annotate(f"{val:.3f}", (val_pos,bar_ax), ha=ha,va=va,size=size,
                        xytext=xytext, textcoords=textcoords)


def plot_coeffs(coeffs, top_n=None,  figsize=(4,5), 
                intercept=False,  intercept_name = 'intercept', 
                annotate=False, ha='left',va='center', size=12, 
                xytext=(4,0), textcoords='offset points'):
    
    """Plots the top_n coefficients from a Series, with optional annotations.
    Source: https://login.codingdojo.com/m/0/13079/101234"""
    
    if (intercept==False) & ( intercept_name in coeffs.index):
        coeffs = coeffs.drop(intercept_name)
        
    if top_n==None:
        
        ## sort all features and set title
        plot_vals = coeffs#.sort_values()
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
    
    if annotate==True:
        annotate_hbars(ax, ha=ha,va=va,size=size,xytext=xytext,
                       textcoords=textcoords)
        
    ## return ax in case want to continue to update/modify figure
    return ax



############################################################################################
############################# FROM: From Regression to Classification ######################
############################################################################################
def evaluate_classification(model, X_train=None,y_train=None,X_test=None,y_test=None,
                            normalize='true',cmap='Blues', label= ': Classification', figsize=(10,5)):
    """Evaluates a classification model using the training data, test data, or both. 

    Args:
        model (Estimator): a fit classification model
        X_train (Frame, optional): X_train data. Defaults to None.
        y_train (Series, optional): y_train data. Defaults to None.
        X_test (_type_, optional): X_test data. Defaults to None.
        y_test (_type_, optional): y_test data. Defaults to None.
        normalize (str, optional): noramlize arg for ConfusionMatrixDisplay. Defaults to 'true'.
        cmap (str, optional): cmap arg for ConfusionMatrixDisplay. Defaults to 'Blues'.
        label (str, optional): label for report header. Defaults to ': Classification'.
        figsize (tuple, optional): figsize for confusion matrix/roc curve subplots. Defaults to (10,5).

    Raises:
        Exception: If neither X_train or X_test is provided. 
    """
    equals = "=="*40
    header="\tCLASSIFICATION REPORT " + label
    dashes='--'*40
    
    # print(f"{dashes}\n{header}\n{dashes}")
    print(f"{equals}\n{header}\n{equals}")
    display(model)
    if (X_train is None) & (X_test is None):
        raise Exception("Must provide at least X_train & y_train or X_test and y_test")
    
    if (X_train is not None) & (y_train is not None):
        ## training data
        header ="[i] Training Data:"
        print(f"{dashes}\n{header}\n{dashes}")
        y_pred_train = model.predict(X_train)
        report_train = metrics.classification_report(y_train, y_pred_train)
        print(report_train)

        fig,ax = plt.subplots(figsize=figsize,ncols=2)
        metrics.ConfusionMatrixDisplay.from_estimator(model,X_train,y_train,
                                                      normalize=normalize, 
                                                      cmap=cmap,ax=ax[0])
        try:
            metrics.RocCurveDisplay.from_estimator(model,X_train,y_train,ax=ax[1])
            ax[1].plot([0,1],[0,1],ls=':')
            ax[1].grid()
        except:
            fig.delaxes(ax[1])
        fig.tight_layout()

        plt.show()

    
        # print(dashes)

        
    if (X_test is not None) & (y_test is not None):
        ## training data
        header = f"[i] Test Data:"
        print(f"{dashes}\n{header}\n{dashes}")
        y_pred_test = model.predict(X_test)
        report_test = metrics.classification_report(y_test, y_pred_test)
        print(report_test)

        fig,ax = plt.subplots(figsize=figsize,ncols=2)
        metrics.ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,
                                                      normalize=normalize, 
                                                      cmap=cmap, ax=ax[0])
        try:
            metrics.RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax[1])
            ax[1].plot([0,1],[0,1],ls=':')
            ax[1].grid()
        except:
            fig.delaxes(ax[1])
        fig.tight_layout()
        plt.show()
        
        
# def evaluate_classification(model, X_train,y_train,X_test,y_test,
#                             normalize='true',cmap='Blues', figsize=(10,5)):
#     "Source: From Regression to Classification Lesson - https://login.codingdojo.com/m/0/13079/101236"
#     header="\tCLASSIFICATION REPORT"
#     dashes='--'*40
#     print(f"{dashes}\n{header}\n{dashes}")
#     ## training data
#     print('[i] Training Data:')
#     y_pred_train = model.predict(X_train)
#     report_train = metrics.classification_report(y_train, y_pred_train)
#     print(report_train)
#     fig,ax = plt.subplots(figsize=figsize,ncols=2)
#     metrics.ConfusionMatrixDisplay.from_estimator(model,X_train,y_train,
#                                                   normalize=normalize,
#                                                    cmap=cmap,ax=ax[0])
#     metrics.RocCurveDisplay.from_estimator(model,X_train,y_train,ax=ax[1])
#     ax[1].plot([0,1],[0,1],ls=':')
#     ax[1].grid()
    
#     fig.tight_layout()
#     plt.show()
     
#     print(dashes)
#     ## test data
#     print(f"[i] Test Data:")
#     y_pred_test = model.predict(X_test)
#     report_test = metrics.classification_report(y_test, y_pred_test)
#     print(report_test)
#     fig,ax = plt.subplots(figsize=figsize,ncols=2)
#     metrics.ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,
#                                                   normalize=normalize,
#                                                    cmap=cmap, ax=ax[0])
                                                   
        
#     metrics.RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax[1])
#     ax[1].plot([0,1],[0,1],ls=':')
#     ax[1].grid()
#     fig.tight_layout()
#     plt.show()

    
    
############################################################################################
############################# FROM: Logistic Regression Coefficients ######################
############################################################################################
"Source: https://login.codingdojo.com/m/0/13079/101237"

def calc_lin_reg(x,coeff=2.713,const = -0.8):
    """Adapted from Source: https://login.codingdojo.com/m/0/13079/101237"""
    return x*coeff + const


def plot_xy(xs,ys):
    """Source: https://login.codingdojo.com/m/0/13079/101237
    Example Use:
    >> xs = np.linspace(-3,3)
    >> ys = calc_lin_reg(xs)
    >> plot_xy(xs,ys)
    """
    plt.plot(xs,ys)
    plt.axvline(0,color='k', zorder=0)
    plt.axhline(0, color='k', zorder=0)

    

def calc_sigmoid(ys):
    """Source: https://login.codingdojo.com/m/0/13079/101237"""
    from math import e
    return 1/(1+e**-ys)


# def get_coeffs_logreg(logreg, feature_names = None, sort=True,ascending=True,
#                       name='LogReg Coefficients', class_index=0):
#     """Source: https://login.codingdojo.com/m/0/13079/101237"""
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
                      as_odds=False):
    """Source: https://login.codingdojo.com/m/0/13079/101237"""

    if feature_names is None:
        feature_names = logreg.feature_names_in_
        
    ## Saving the coefficients
    coeffs = pd.Series(logreg.coef_[class_index],
                       index= feature_names, name=name)
    
    # use .loc to add the intercept to the series
    coeffs.loc['intercept'] = logreg.intercept_[class_index]
        
    if as_odds==True:
        coeffs = np.exp(coeffs)
    if sort == True:
        coeffs = coeffs.sort_values(ascending=ascending)
    
        
    return coeffs



############################################################################################
############################# FROM: (Optional) Advanced MatPlotLib ######################
############################################################################################
"Source: https://login.codingdojo.com/m/0/13079/101258"

def get_colors_gt_lt(coeffs, threshold=1, color_lt ='darkred',
                     color_gt='forestgreen', color_else='gray'):
    """Creates a dictionary of features:colors based on if value is > or < threshold
    Source: https://login.codingdojo.com/m/0/13079/101258
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



def plot_coeffs_color(coeffs, top_n=None,  figsize=(8,6), color_dict=None,
                   plot_kws = {} ):
    """Plots series of coefficients
    Source: https://login.codingdojo.com/m/0/13079/101258
    
    Args:
        ceoffs (pands Series): importance values to plot
        top_n (int): The # of features to display (Default=None).
                            If None, display all.
                        otherwise display top_n most important

        figsize (tuple): figsize tuple for .plot
        color_dict (dict): dict with index values as keys with color to use as vals
                            Uses series.index.map(color_dict).
        plot_kws (dict): additional keyword args accepted by panda's .plot
        
    Returns:
        Axis: matplotlib axis
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
    if color_dict is not None:
        ## Getting color list and saving to plot_kws
        colors = plot_vals.index.map(color_dict)
        plot_kws = plot_kws.update({'color':colors})
    
    
    ax = plot_vals.plot(kind='barh', figsize=figsize,**plot_kws)
    ax.set(xlabel='Coefficient',
            ylabel='Feature Names',
            title=title)
    
    ## return ax in case want to continue to update/modify figure
    return ax




def evaluate_regression(model, X_train,y_train, X_test, y_test, as_frame=True): 
    """Evaluates a scikit learn regression model using r-squared and RMSE
    if as_frame = True, displays results as dataframe.
    """
    
    ## Training Data
    y_pred_train = model.predict(X_train)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    rmse_train = metrics.mean_squared_error(y_train, y_pred_train, 
                                            squared=False)
    mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
    

    ## Test Data
    y_pred_test = model.predict(X_test)
    r2_test = metrics.r2_score(y_test, y_pred_test)
    rmse_test = metrics.mean_squared_error(y_test, y_pred_test, 
                                            squared=False)
    mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
    
    if as_frame:
        df_version =[['Split','R^2','MAE','RMSE']]
        df_version.append(['Train',r2_train, mae_train, rmse_train])
        df_version.append(['Test',r2_test, mae_test, rmse_test])
        df_results = pd.DataFrame(df_version[1:], columns=df_version[0])
        df_results = df_results.round(2)
        display(df_results.style.hide(axis='index').format(precision=2, thousands=','))
        
    else: 
        print(f"Training Data:\tR^2 = {r2_train:,.2f}\tRMSE = {rmse_train:,.2f}\tMAE = {mae_train:,.2f}")
        print(f"Test Data:\tR^2 = {r2_test:,.2f}\tRMSE = {rmse_test:,.2f}\tMAE = {mae_test:,.2f}")


        
        

## ADMIN VERSION 
def evaluate_classification_admin(model, X_train=None,y_train=None,X_test=None,y_test=None,
                            normalize='true',cmap='Blues', label= ': (Admin)', figsize=(10,5)):
    header="\tCLASSIFICATION REPORT " + label
    dashes='--'*40
    print(f"{dashes}\n{header}\n{dashes}")
    
    if (X_train is None) & (X_test is None):
        raise Exception("Must provide at least X_train & y_train or X_test and y_test")
    
    if (X_train is not None) & (y_train is not None):
        ## training data
        print(f"[i] Training Data:")
        y_pred_train = model.predict(X_train)
        report_train = metrics.classification_report(y_train, y_pred_train)
        print(report_train)

        fig,ax = plt.subplots(figsize=figsize,ncols=2)
        metrics.ConfusionMatrixDisplay.from_estimator(model,X_train,y_train,
                                                      normalize=normalize, 
                                                      cmap=cmap,ax=ax[0])
        try:
            metrics.RocCurveDisplay.from_estimator(model,X_train,y_train,ax=ax[1])
            ax[1].plot([0,1],[0,1],ls=':')
            ax[1].grid()
        except:
            fig.delaxes(ax[1])
        fig.tight_layout()

        plt.show()

    
        print(dashes)

        
    if (X_test is not None) & (y_test is not None):
        ## training data
        print(f"[i] Test Data:")
        y_pred_test = model.predict(X_test)
        report_test = metrics.classification_report(y_test, y_pred_test)
        print(report_test)

        fig,ax = plt.subplots(figsize=figsize,ncols=2)
        metrics.ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,
                                                      normalize=normalize, 
                                                      cmap=cmap, ax=ax[0])
        try:
            metrics.RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax[1])
            ax[1].plot([0,1],[0,1],ls=':')
            ax[1].grid()
        except:
            fig.delaxes(ax[1])
        fig.tight_layout()
        plt.show()
        