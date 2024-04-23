
"""A collection of convenient csv urls and sklearn datasets as dataframes."""
def load_data(*args,**kwargs):
    raise Exception('load_data() has been replaced by individual load functions. i.e. fs.datasets.load_boston()')



def read_csv_from_url(url,verbose=False,read_csv_kwds={}):
    """Loading function to load all .csv datasets.
    Args:
        url (str): csv raw link
        verbose (bool): Controls display of loading message and .head()
        read_csv_kwds (dict): dict of commands to feed to pd.read_csv()
    Returns:
        df (DataFrame): the dataset("""
    import pandas as pd
    from IPython.display import display
    ## Load and return dataset
    # if verbose: 
        # print(f"[i] Loading {dataset} from url:\n{url}")
    if read_csv_kwds is not None:
        df = pd.read_csv(url,**read_csv_kwds)
    else:
        df = pd.read_csv(url)
    if verbose:
        display(df.head())
    return df


def load_superhero_info(verbose=False,read_csv_kwds={}):

    url = 'https://raw.githubusercontent.com/jirvingphd/dsc-data-cleaning-project-online-ds-ft-100719/master/heroes_information.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)
    

def load_superhero_powers(verbose=False,read_csv_kwds={}):
    url = "https://raw.githubusercontent.com/learn-co-students/dsc-data-cleaning-project-online-ds-ft-100719/master/super_hero_powers.csv"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)

def load_titanic(verbose=False,kaggle=False,read_csv_kwds={}):
    if kaggle:
        url="https://raw.githubusercontent.com/jirvingphd/fsds/master/datafiles/titanic.csv.gz"
        if verbose:
            print('[i] For dataset details, see https://www.kaggle.com/heptapod/titanic')
    else:
        url ="https://raw.githubusercontent.com/jirvingphd/dsc-dealing-missing-data-lab-online-ds-ft-100719/master/titanic.csv"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_mod1_proj(verbose=False,read_csv_kwds={}):
    url = "https://raw.githubusercontent.com/learn-co-students/dsc-v2-mod1-final-project-online-ds-ft-100719/master/kc_house_data.csv"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)
        

def load_population(verbose=False,read_csv_kwds={}):
    url = "https://raw.githubusercontent.com/learn-co-students/dsc-subplots-and-enumeration-lab-online-ds-ft-100719/master/population.csv"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)

def load_autompg(verbose=True,read_csv_kwds={}):
    
    if verbose:
        print('[i] Source url with details: https://www.kaggle.com/uciml/autompg-dataset')
    
    url = 'https://raw.githubusercontent.com/jirvingphd/dsc-dealing-with-categorical-variables-online-ds-ft-100719/master/auto-mpg.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)




def load_boston(verbose=False):
    
    ## Load Sklearn Datasets
    from sklearn import datasets
    import pandas as pd 
    
    if verbose:
        print("[i] Loading boston housing dataset from sklearn.datasets")
    ## load data dict
    data_dict =  datasets.load_boston()
    # load features
    df_features = pd.DataFrame(data_dict['data'],columns=data_dict['feature_names'])
    # load targets]
    df_features['price'] =data_dict['target']
    
    ## Dropping Blacks column
    df_features.drop('B',axis=1,inplace=True)
    descr = data_dict['DESCR'].split('\n')
    descr = [line for line in descr if "- B" not in line ]
    
    # set output df
    df = df_features
    if verbose:
        print("\n".join(descr))
    
    return df 



def load_iris(verbose=False):
    from sklearn import datasets
    import pandas as pd
    if verbose:
        print('[i] Loading iris datset from sklearn.datasets')
    data_dict =  datasets.load_iris()
    
    # Get dataframe
    df_features = pd.DataFrame(data_dict['data'],columns=data_dict['feature_names'])
    df_features['target'] = data_dict['target']


    # Get mapper for target names
    iris_map = dict(zip( 
        list(set(data_dict['target'])),
        data_dict['target_names'])
                )
    df_features['target_name']=df_features['target'].map(iris_map)
    df = df_features
    if verbose:
        print(data_dict['DESCR'])   
    return df


def load_ames_train(verbose=False,subset=False, read_csv_kwds={}):
    """Loads height vs weight dataset"""
    import requests 
    if verbose:
        res = requests.get('https://raw.githubusercontent.com/learn-co-students/dsc-project-eda-with-pandas-onl01-dtsc-pt-041320/master/data_description.txt')
        info = res.text
        print(info)
    url='https://raw.githubusercontent.com/learn-co-students/dsc-project-eda-with-pandas-onl01-dtsc-pt-041320/master/ames_train.csv'
    
    df  = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)
    
    if subset:
        subset = ['YrSold', 'MoSold', 'Fireplaces', 'TotRmsAbvGrd', 'GrLivArea',
          'FullBath', 'YearRemodAdd', 'YearBuilt', 'OverallCond', 'OverallQual', 'LotArea', 'SalePrice']
        df = df[subset]
    return  df


def load_ames_test(verbose=False,read_csv_kwds={}):
    import requests 
    if verbose:
        res = requests.get('https://raw.githubusercontent.com/learn-co-students/dsc-project-eda-with-pandas-onl01-dtsc-pt-041320/master/data_description.txt')
        info = res.text
        print(info)
    """Loads height vs weight dataset"""
    url='https://raw.githubusercontent.com/learn-co-students/dsc-project-eda-with-pandas-onl01-dtsc-pt-041320/master/ames_test.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)



def load_height_weight(verbose=False,read_csv_kwds={}):
    """Loads height vs weight dataset"""
    url='https://raw.githubusercontent.com/jirvingphd/dsc-probability-density-function-online-ds-ft-100719/master/weight-height.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_iowa_prisoners(verbose=False,vers='raw',read_csv_kwds={}):
    import pandas as pd
    if 'raw' in vers:
        url ='https://raw.githubusercontent.com/jirvingphd/iowa-prisoner-recidivism-mod-3-project/b6a92d1474c3eee790ab894f79751d69578bfb18/datasets/FULL_3-Year_Recidivism_for_Offenders_Released_from_Prison_in_Iowa.csv'
    else:
        url = 'https://raw.githubusercontent.com/jirvingphd/iowa-prisoner-recidivism-mod-3-project/b6a92d1474c3eee790ab894f79751d69578bfb18/datasets/Iowa_Prisoners_Renamed_Columns_fsds_100719.csv'
    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)
    #pd.set_option('display.precision',3)
    return df

def load_height_by_country(verbose=False,read_csv_kwds={}):
    url='https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/height_by_country_age18.csv'
    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)

    if verbose:
        source="http://ncdrisc.org/data-downloads-height.html"
        print(f'Source of dataset: {source}')
        
    return df

def load_yields(verbose=False,version='full',read_csv_kwds=dict(sep=r'\s+', index_col=0)):
    """Loads dataset from Polynomial Regression readme"""
    
    if version=='full':
        url = 'https://raw.githubusercontent.com/jirvingphd/dsc-bias-variance-trade-off-online-ds-pt-100719/master/yield2.csv'
    else:
        url='https://raw.githubusercontent.com/jirvingphd/dsc-polynomial-regression-online-ds-pt-100719/master/yield.csv'

    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)
    
    return df


### TIME SERIES

# baltimore_crime ="https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/BPD_Part_1_Victim_Based_Crime_Data.csv"
# std_rates = "https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/STD%20Cases.csv"
# no_sex_xlsx = "https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/Americans%20Sex%20Frequency.xlsx"

# learn_passengers="https://raw.githubusercontent.com/learn-co-students/dsc-removing-trends-lab-online-ds-ft-100719/master/passengers.csv"

def load_ts_baltimore_crime_full(read_csv_kwds={}):
    url ="https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/BPD_Part_1_Victim_Based_Crime_Data.csv"
    return  read_csv_from_url(url, verbose=False,read_csv_kwds=read_csv_kwds)

### TIME SERIES
def load_ts_baltimore_crime_counts(read_csv_kwds={}):
    url="https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/baltimore_crime_counts_2014-2019.csv"
    return  read_csv_from_url(url, verbose=False,read_csv_kwds=read_csv_kwds)


def load_ts_mintemp(verbose=False,read_csv_kwds={}):
    """Loads min_temp.csv from """
    if verbose:
        print("From Introduction to Time Series")
    url='https://raw.githubusercontent.com/jirvingphd/dsc-introduction-to-time-series-online-ds-ft-100719/master/min_temp.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_ts_nyse_monthly(verbose=False,read_csv_kwds={}):
    """Loads NYSE_.csv from """
    if verbose:
        print("From Introduction to Time Series")
    url='https://raw.githubusercontent.com/jirvingphd/dsc-introduction-to-time-series-online-ds-ft-100719/master/NYSE_monthly.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_ts_exch_rates(verbose=False,read_csv_kwds={}):
    # if verbose:
    url="https://raw.githubusercontent.com/jirvingphd/dsc-basic-time-series-models-online-ds-ft-100719/master/exch_rates.csv"
    return read_csv_from_url(url, verbose=verbose, read_csv_kwds=read_csv_kwds)


def load_ts_google_trends(read_csv_kwds={}):
    url='https://raw.githubusercontent.com/jirvingphd/dsc-corr-autocorr-in-time-series-online-ds-ft-100719/master/google_trends.csv'
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)


def load_ts_winning_400m(read_csv_kwds={}):
    url="https://raw.githubusercontent.com/jirvingphd/dsc-arma-models-lab-online-ds-ft-100719/master/winning_400m.csv"
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)


def load_ts_std_cases(read_csv_kwds={}):
    url = 'https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/STD%20Cases.csv'
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)

def load_ts_american_sex_frequency(read_csv_kwds={}):
    url = 'https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/Americans%20Sex%20Frequency.xlsx'
    import pandas as pd
    
    return pd.read_excel(url,**read_csv_kwds)
    # return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)

# def load_ts_co2(read_csv_kwds={}):
#     import statsmodels.api as sm
#     df = sm.datasets.co2.load()
#     return df


def load_AB_multiple_choice(verbose=False,read_csv_kwds={}):
    url='https://raw.githubusercontent.com/jirvingphd/dsc-in-depth-ab-testing-lab-online-ds-pt-100719/master/multipleChoiceResponses_cleaned.csv'
    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)

    if verbose:
        from IPython.display import display
        display(df.head())
        
    return df

def load_AB_homepage_actions(verbose=False,read_csv_kwds={}):
    url="https://raw.githubusercontent.com/jirvingphd/dsc-website-ab-testing-lab-online-ds-pt-100719/master/homepage_actions.csv"
    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)

    if verbose:
        from IPython.display import display
        display(df.head())
        
    return df



def load_stock_df(**kwargs):
    import pandas as pd
    url ='https://raw.githubusercontent.com/jirvingphd/capstone-project-using-trumps-tweets-to-predict-stock-market/master/data/SP500_1min_01_23_2020_full.xlsx'
    stock_df = pd.read_excel(url,**kwargs)
    return stock_df
    

def load_heart_disease(verbose=False,read_csv_kwds={}):
    import pandas as pd
    url = "https://raw.githubusercontent.com/jirvingphd/dsc-gaussian-naive-bayes-lab-online-ds-pt-100719/solution/heart.csv"
    return read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_spam(verbose=False,read_csv_kwds=dict( sep='\t', names=['label', 'text'])):
    url = "https://raw.githubusercontent.com/jirvingphd/dsc-document-classification-with-naive-bayes-lab-online-ds-pt-100719/solution/SMSSpamCollection"
    return read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)




def load_tennis(read_csv_kwds={}):
    url="https://raw.githubusercontent.com/jirvingphd/dsc-decision-trees-with-sklearn-codealong-online-ds-pt-100719/master/tennis.csv"
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)



def load_nlp_finding_trump(read_csv_kwds={}):
    url='https://raw.githubusercontent.com/jirvingphd/online-ds-pt-1007109-text-classification-finding-trump/master/finding-trump.csv'
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)

def load_nlp_trump_tweets(read_csv_kwds={}):
    url='https://raw.githubusercontent.com/jirvingphd/capstone-project-using-trumps-tweets-to-predict-stock-market/master/data/trump_tweets_12012016_to_01012020.csv'
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)



####### NEW 08-31-2021 ###

def load_pokemon(read_csv_kwds={"index_col":0}):
    url="https://raw.githubusercontent.com/jirvingphd/fsds/master/datafiles/pokemon/pokemon_alopez247.csv"
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)

def load_hotel_bookings(read_csv_kwds={"encoding":'latin-1'}):
    url='https://raw.githubusercontent.com/jirvingphd/fsds/master/datafiles/hotel_bookings.csv.gz'
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)

def load_videogame_sales(read_csv_kwds={}):
    url="https://raw.githubusercontent.com/jirvingphd/fsds/master/datafiles/vgsales.csv"
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)





def load_king_county_housing(verbose=False,project_vers=True, read_csv_kwds={}):
    if project_vers==True:
        if verbose:
            print('[i] Loading the project-version of the dataset.')
        url = "https://github.com/jirvingphd/fsds/raw/master/datafiles/king-county-housing-project-version.csv.gz"
    else:
        if verbose:
            print('[i] Loading the kaggle-version of the dataset.')
        url = "https://github.com/jirvingphd/fsds/raw/master/datafiles/king-county-housing-kaggle-version.csv.gz"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)
        