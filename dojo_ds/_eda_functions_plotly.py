###### PLOTLY VERSIONS

def set_template(template='seaborn'):
    """
    Sets the default template for Plotly plots.

    Parameters:
    - template (str): The name of the template to set. Default is 'seaborn'.
    """
    import plotly.io as pio
    pio.templates.default = template
    

def plotly_explore_numeric(df, x, width=1000, height=500):
    """
    Generate a histogram with a box plot overlay to explore the distribution of a numeric variable.

    Parameters:
    - df: DataFrame - The input DataFrame containing the data.
    - x: str - The name of the column representing the numeric variable to be explored.
    - width: int, optional - The width of the generated plot in pixels. Default is 1000.
    - height: int, optional - The height of the generated plot in pixels. Default is 500.

    Returns:
    - fig: plotly.graph_objects.Figure - The generated plotly figure object.
    """
    import plotly.express as px
    fig = px.histogram(df, x=x, marginal='box', title=f'Distribution of {x}', 
                       width=width, height=height)
    return fig


def plotly_explore_categorical(df, x ,width=1000, height=500):
    """
    Generate a histogram plot using Plotly to explore the distribution of a categorical variable in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - x (str): The name of the categorical variable to explore.
    - width (int, optional): The width of the plot in pixels. Default is 1000.
    - height (int, optional): The height of the plot in pixels. Default is 500.

    Returns:
    - fig (plotly.graph_objects.Figure): The generated histogram plot.
    """
    import plotly.express as px
    fig = px.histogram(df,x=x,color=x,title=f'Distribution of {x}', 
                         width=width, height=height)
    return fig
    

def plotly_numeric_vs_target(df, x, y='SalePrice', trendline='ols', add_hoverdata=True,
                            width=800, height=600):
    """
    Creates a scatter plot using Plotly to visualize the relationship between a numeric variable and a target variable.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - x (str): The name of the numeric variable to be plotted on the x-axis.
    - y (str): The name of the target variable to be plotted on the y-axis. Default is 'SalePrice'.
    - trendline (str): The type of trendline to be added to the plot. Default is 'ols' (ordinary least squares).
    - add_hoverdata (bool): Whether to include hover data in the plot. Default is True.
    - width (int): The width of the plot in pixels. Default is 800.
    - height (int): The height of the plot in pixels. Default is 600.

    Returns:
    - pfig (plotly.graph_objects.Figure): The Plotly figure object representing the scatter plot.
    """
    if add_hoverdata == True:
        hover_data = list(df.drop(columns=[x, y]).columns)
    else: 
        hover_data = None
    import plotly.express as px
    pfig = px.scatter(df, x=x, y=y, hover_data=hover_data,
                      trendline=trendline, trendline_color_override='red',
                      title=f"{x} vs. {y}", width=width, height=height)

    pfig.update_traces(marker=dict(size=3), line=dict(dash='dash'))
    return pfig


def plotly_numeric_vs_target(df, x, y='SalePrice', trendline='ols', add_hoverdata=True,
                            width=800, height=600):
    """
    Creates a scatter plot using Plotly to visualize the relationship between a numeric variable and a target variable.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - x (str): The name of the numeric variable to be plotted on the x-axis.
    - y (str, optional): The name of the target variable to be plotted on the y-axis. Default is 'SalePrice'.
    - trendline (str, optional): The type of trendline to be added to the plot. Default is 'ols' (ordinary least squares).
    - add_hoverdata (bool, optional): Whether to include hover data in the plot. Default is True.
    - width (int, optional): The width of the plot in pixels. Default is 800.
    - height (int, optional): The height of the plot in pixels. Default is 600.

    Returns:
    - pfig (plotly.graph_objects.Figure): The Plotly figure object representing the scatter plot.
    """
    if add_hoverdata == True:
        hover_data = list(df.drop(columns=[x, y]).columns)
    else: 
        hover_data = None
    import plotly.express as px
    pfig = px.scatter(df, x=x, y=y, hover_data=hover_data,
                      trendline=trendline, trendline_color_override='red',
                      title=f"{x} vs. {y}", width=width, height=height)

    pfig.update_traces(marker=dict(size=3), line=dict(dash='dash'))
    return pfig
def plotly_numeric_vs_target(df, x, y='SalePrice', trendline='ols',add_hoverdata=True,
                            width=800, height=600,):
    if add_hoverdata == True:
        hover_data = list(df.drop(columns=[x,y]).columns)
    else: 
        hover_data = None
    import plotly.express as px
    pfig = px.scatter(df, x=x, y=y,# template='plotly_dark', 
                     hover_data=hover_data,
                      trendline=trendline,
                      trendline_color_override='red',
                     title=f"{x} vs. {y}",
                      width=width, height=height)

    
    pfig.update_traces(marker=dict(size=3),
                      line=dict(dash='dash'))
    return pfig
    

# def plotly_categorical_vs_target(df, x, y='SalePrice', agg='mean',width=800,height=500,):
#     if agg=='mean':
#         plot_df = df.groupby(x,as_index=False)[y].mean().sort_values(y, ascending=False)
        
#     elif agg=='median':
#         plot_df = df.groupby(x,as_index=False)[y].median().sort_values(y,ascending=False)
        
#     else:
#         plot_df = df
        
#     fig = px.bar(plot_df, x=x,y=y, color=x, title=f'Compare {agg.title()} {y} by {x}',
#                  width=width, height=height)

                
#     return fig

def plotly_categorical_vs_target(df, x, y='SalePrice', histfunc='avg', width=800,height=500):
    """
    Plots a categorical variable against a target variable using Plotly.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - x (str): The name of the categorical variable.
    - y (str): The name of the target variable. Default is 'SalePrice'.
    - histfunc (str): The aggregation function to use for the histogram. Default is 'avg'.
    - width (int): The width of the plot. Default is 800.
    - height (int): The height of the plot. Default is 500.

    Returns:
    - fig (plotly.graph_objects.Figure): The Plotly figure object.
    """
    import plotly.express as px
    fig = px.histogram(df, x=x,y=y, color=x, width=width, height=height,
                       histfunc=histfunc, title=f'Compare {histfunc.title()} {y} by {x}')
    fig.update_layout(showlegend=False)
    return fig


def plotly_correlation(df, cmap='magma', cols=None):
    """
    Generates a correlation heatmap using Plotly.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    cmap (str, optional): The color map to use for the heatmap. Defaults to 'magma'.
    cols (list, optional): The columns to include in the correlation calculation. If None, all columns are included. Defaults to None.

    Returns:
    plotly.graph_objects.Figure: The correlation heatmap figure.
    """
    import plotly.express as px

    if cols == None:
        cols = df.columns
    corr = df[cols].corr(numeric_only=True).round(2)
    fig = px.imshow(corr, text_auto=True,width=600, height=600,
                    color_continuous_scale=cmap,
                    color_continuous_midpoint=0, 
                   title='Correlation Heatmap')
    return fig


# if __name__ == "__main__":
#     set_template()