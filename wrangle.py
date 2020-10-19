# Importing libraries:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import numpy as np
import scipy as sp 
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Importing the os library specifically for reading the csv once I've created the file in my working directory.
import os
import env

# Setting up the user credentials:

from env import host, user, password

# The 
def get_db(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}' 

# acquire

def get_zillow_data():

    '''
    This function will allow the user to retrieve all tables from the Zillow database from the Codeup DB source. 
    It will acquire the data, import it as a dataframe, and then write that dataframe to a .csv file in the local directory.
    '''

    zillow_sql_query = '''
                SELECT * 
                    FROM properties_2017
                    JOIN (select id, logerror, pid, tdate FROM predictions_2017 pred_2017
                    JOIN (SELECT parcelid AS pid, Max(transactiondate) as tdate FROM predictions_2017 GROUP BY parcelid) AS sq1
                    ON (pred_2017.parcelid = sq1.pid AND pred_2017.transactiondate = sq1.tdate)) AS sq2
                    ON (properties_2017.parcelid = sq2.pid)
                    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
                    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
                    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
                    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
                    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
                    LEFT JOIN storytype USING (storytypeid)
                    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
                    LEFT JOIN unique_properties USING (parcelid)
                    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
                '''
    
    
    filename = 'zillow_clustering_data.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        zillow_df = pd.read_sql(zillow_sql_query, get_db('zillow'))
        zillow_df.to_csv(filename, index = False)
        
    return zillow_df


# Dropping un-needed columns:
def drop_cols(df):
    '''
    Use this function if the zillow dataframe has extra id columns after using the sql acquire functions.
    '''
    df.drop(columns = ['fullbathcnt', 'parcelid', 'id.1', 'id', 'pid', 'propertyzoningdesc', "propertycountylandusecode", 'heatingorsystemtypeid', 'regionidcity', 'regionidcounty', 'unitcnt','propertylandusedesc', 'rawcensustractandblock', 'finishedsquarefeet12', 'calculatedbathnbr', 'heatingorsystemdesc', 'buildingqualitytypeid', 'regionidzip', 'assessmentyear', 'tdate', 'censustractandblock'], inplace = True)
    return df

def prep_data(df, id_list):
    '''
    In order to use this function, you will need to create a list of property use type ids (if using the Zillow data),
    and use that variable (which is a list), inside the 2nd argument of this function.
    The first argument is the dataframe, the 2nd argument is a list of property use ids that you want to use going forward.
    The function will remove all property use ids that are not in the `id_list` argument list.
    This function will effectively reduce the number of rows, not removing any columns.
    '''
    # Taking out these rows and columns as well.
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
    # df.unitcnt = df.unitcnt.fillna(1)
    df.latitude = df.latitude / 1_000_000
    df.longitude = df.longitude / 1_000_000


    # changing the year to an int.
    # df["yearbuilt"] = df["yearbuilt"].astype('int')

    # for example: id_list = [261.0, 260.0, 262.0, 263.0, 264.0]
    df = df[df.propertylandusetypeid.isin(id_list)]
    return df


# Handling missing values:
def handle_missing_values(df, col_limit = .5, row_limit = .5):

    # df.drop(columns = ['id.1', 'id', 'pid', 'propertyzoningdesc', 'calculatedbathnbr', 'heatingorsystemdesc'], inplace = True)
    # Setting the threshold for columns to drop:
    col_thresh = int(round(col_limit * len(df.index), 0))
    df.dropna(axis = 1, thresh = col_thresh, inplace = True)
    # Now for the rows:
    row_thresh = int(round(col_limit * len(df.columns), 0))
    df.dropna(axis = 0, thresh = row_thresh, inplace = True)
    return df

# Function for imputing remaining missing values occurs after train, validate, test split of data:

def split_zillow_data(df):
    # df = get_mall_data()
    # Splitting my data based on the target variable of tenure:
    train_validate, test = train_test_split(df, test_size=.15, random_state=123)
    
    # Splitting the train_validate set into the separate train and validate datasets.
    train, validate = train_test_split(train_validate, test_size=.20, random_state=123)
    
    # Printing the shape of each dataframe:
    print(f'Shape of train df: {train.shape}')
    print(f'Shape of validate df: {validate.shape}')
    print(f'Shape of test df: {test.shape}')
    return train, validate, test

# Impute missing values, after splitting.
# These columns make most sense to replace missing values with the median value.
# This function should take care of all missing values at one time once I've split the data. It expects three dataframes as input, and returns 3 dataframes out.

def impute_missing_values_all(train, validate, test):
    
    
    
    cols = [
    "structuretaxvaluedollarcnt",
    "taxamount",
    "taxvaluedollarcnt",
    "landtaxvaluedollarcnt",
    "structuretaxvaluedollarcnt",
    "calculatedfinishedsquarefeet",
    "lotsizesquarefeet",
    "age"
    ]


    for col in cols:
        median = train[col].median()
        train[col].fillna(median, inplace=True)
        validate[col].fillna(median, inplace=True)
        test[col].fillna(median, inplace=True)


    # Categorical/Discrete columns to use mode to replace nulls

    cols2 = [
        "yearbuilt",
    ]

    for col in cols2:
        mode = int(train[col].mode()) # I had some friction when this returned a float (and there were no decimals anyways)
        train[col].fillna(value=mode, inplace=True)
        validate[col].fillna(value=mode, inplace=True)
        test[col].fillna(value=mode, inplace=True)
    
    # Taking care of cols using mean.
    cols3 = [
        "acres",
        "structure_dollar_per_sqft",
        "land_dollar_per_sqft",
        "taxrate"
    ]

    for col in cols3:
        train[col].fillna(value=train[col].mean(), inplace=True)
        validate[col].fillna(value=validate[col].mean(), inplace=True)
        test[col].fillna(value=test[col].mean(), inplace=True)

    
    # cols4 = ["heatingorsystemdesc"]

    # for col in cols4:
    #     #median = train[col].median()
    #     train[col].fillna("None", inplace = True)
    #     validate[col].fillna("None", inplace = True)
    #     test[col].fillna("None", inplace = True)
    
        
    return train, validate, test


# Complete wrangle function:

# Starting with acquire, and ending with scaling the values:

# Build the entire wrangle function for zillow:

def wrangle_zillow():
    
    # acquiring the dataframe:
    df = get_zillow_data
    
    # Dropping un-needed columns:
    # def drop_these_cols(df):
    #     '''
    #     Use this function if the zillow dataframe has extra id and description columns after using the sql acquire functions.
    #     '''
    #     df = df.drop(columns = ['id.1', 'id', 'heatingorsystemdesc', 'propertylandusedesc'])
    #     return df
    
    # This next section of code allows the function to focus only on single unit properties
    id_list = [261.0, 260.0, 262.0, 263.0, 264.0]
    
    def property_type_focus(df, id_list):
        '''
        In order to use this function, you will need to create a list of property use type ids (if using the Zillow data),
        and use that variable (which is a list), inside the 2nd argument of this function.
        The first argument is the dataframe, the 2nd argument is a list of property use ids that you want to use going forward.
        The function will remove all property use ids that are not in the `id_list` argument list.
        '''

        df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
        df.unitcnt = df.unitcnt.fillna(1)
        # for example: id_list = [261.0, 260.0, 262.0, 263.0, 264.0]
        df = df[df.propertylandusetypeid.isin(id_list)]

        return df

    # Handling missing values. All I need to input is the dataframe (df), the rest of the parameters are already set.
    def handle_missing_values(df, col_limit = .6, row_limit = .6):

        # Setting the threshold for columns to drop:
        col_thresh = int(round(col_limit * len(df.index), 0))
        df.dropna(axis = 1, thresh = col_thresh, inplace = True)
        # Now for the rows:
        row_thresh = int(round(col_limit * len(df.columns), 0))
        df.dropna(axis = 0, thresh = row_thresh, inplace = True)
        return df
    
    # Function for imputing remaining missing values occurs after train, validate, test split of data:

    def split_zillow_data(df):
        # df = get_mall_data()
        # Splitting my data based on the target variable of tenure:
        train_validate, test = train_test_split(df, test_size=.15, random_state=123)

        # Splitting the train_validate set into the separate train and validate datasets.
        train, validate = train_test_split(train_validate, test_size=.20, random_state=123)

        # Printing the shape of each dataframe:
        print(f'Shape of train df: {train.shape}')
        print(f'Shape of validate df: {validate.shape}')
        print(f'Shape of test df: {test.shape}')
        return train, validate, test
    
    # Impute missing values, after splitting.
    # This function should take care of all missing values at one time once I've split the data. It expects three dataframes as input, and returns 3 dataframes out.

    def imputing_missing_values_all(train, validate, test):

        cols = [
        "structuretaxvaluedollarcnt",
        "taxamount",
        "taxvaluedollarcnt",
        "landtaxvaluedollarcnt",
        "structuretaxvaluedollarcnt",
        "finishedsquarefeet12",
        "calculatedfinishedsquarefeet",
        "fullbathcnt",
        "lotsizesquarefeet",
        # "heatingorsystemtypeid"
        ]


        for col in cols:
            median = train[col].median()
            train[col].fillna(median, inplace=True)
            validate[col].fillna(median, inplace=True)
            test[col].fillna(median, inplace=True)


        # Categorical/Discrete columns to use mode to replace nulls

        cols2 = [
            "buildingqualitytypeid",
            "regionidcity",
            "regionidzip",
            "yearbuilt",
            "regionidcity",
            # "censustractandblock"
        ]

        for col in cols2:
            mode = int(train[col].mode()) # I had some friction when this returned a float (and there were no decimals anyways)
            train[col].fillna(value=mode, inplace=True)
            validate[col].fillna(value=mode, inplace=True)
            test[col].fillna(value=mode, inplace=True)

        # Taking care of unit count.
        # cols3 = [
        #     "unitcnt"
        # ]

        # for col in cols3:
        #     train[col].fillna(value=1, inplace=True)
        #     validate[col].fillna(value=1, inplace=True)
        #     test[col].fillna(value=1, inplace=True)


        # cols4 = ["heatingorsystemdesc"]

        # for col in cols4:
        #     #median = train[col].median()
        #     train[col].fillna("None", inplace = True)
        #     validate[col].fillna("None", inplace = True)
        #     test[col].fillna("None", inplace = True)


    return train, validate, test
    print(train.shape)
    print(validate.shape)
    print(test.shape)


# Dealing with outliers
def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
    #                 for col in df.select_dtypes('number')}
    # return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)

    return df






def imputing_missing_values(df):
    
    # First, inputing the median values:
    df.regionidcity = df.regionidcity.fillna(df.regionidcity.median())
    df.regionidzip = df.regionidzip.fillna(df.regionidzip.median())
    df.yearbuilt = df.yearbuilt.fillna(df.yearbuilt.median())
    df.censustractandblock = df.censustractandblock.fillna(df.censustractandblock.median())
    
    # Now using the mean to input the rest of the missing values:
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.mean())
    df.finishedsquarefeet12 = df.finishedsquarefeet12.fillna(df.finishedsquarefeet12.mean())
    df.calculatedbathnbr = df.calculatedbathnbr.fillna(df.calculatedbathnbr.mean())
    df.fullbathcnt = df.fullbathcnt.fillna(round(df.fullbathcnt.mean(),0))
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(round(df.calculatedfinishedsquarefeet.mean(),0))
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(round(df.structuretaxvaluedollarcnt.mean(),0))
    df.taxamount = df.taxamount.fillna(round(df.taxamount.mean(),0))
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.fillna(round(df.landtaxvaluedollarcnt.mean(),0))
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(round(df.taxvaluedollarcnt.mean(),0))


print("Loaded zillow wrangle functions successfully.")