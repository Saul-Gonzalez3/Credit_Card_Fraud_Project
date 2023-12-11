# imports: 
import numpy as np
import pandas as pd
import os 

# spliting data
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import MinMaxScaler
    
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
#imports for regression modeling

from sklearn.linear_model import LogisticRegression
# import K Nearest neighbors:
from sklearn.neighbors import KNeighborsClassifier
# import Decision Trees:
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
# import Random Forest:
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

# interpreting our models:
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#feature selection
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from sklearn.linear_model import LinearRegression

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

"""
*-----------------------*
|                       |
|       FUNCTIONS       |
|                       |
*-----------------------*
"""
#----------------Data Preperation------------------#
def identify_columns(df):
    cat_cols, num_cols = [], []

    for col in df.columns:
        if df[col].dtype == 'O':
            cat_cols.append(col)
        else:
            if df[col].nunique() < 10:
                cat_cols.append(col)
            else:
                num_cols.append(col)

    return cat_cols, num_cols

#----------Split Function--------#
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"""
    train -> {train.shape}
    validate -> {validate.shape}
    test -> {test.shape}""")
    
    return train, validate, test

#----------Min Max Scaler--------#
def scale_fraud_data(train, validate, test):
    
    cols = ['fraud_bool', 'income', 'name_email_similarity',
       'prev_address_months_count', 'current_address_months_count',
       'customer_age', 'days_since_request', 'intended_balcon_amount',
       'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w',
       'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
       'credit_risk_score', 'email_is_free', 'phone_home_valid',
       'phone_mobile_valid', 'bank_months_count', 'has_other_cards',
       'proposed_credit_limit', 'foreign_request', 'session_length_in_minutes',
       'keep_alive_session', 'device_distinct_emails_8w', 'device_fraud_count',
       'month', 'payment_type_encoded', 'employment_status_encoded',
       'housing_status_encoded', 'device_os_encoded', 'payment_type_AB',
       'payment_type_AC', 'payment_type_AD', 'payment_type_AE',
       'employment_status_CB', 'employment_status_CC', 'employment_status_CD',
       'employment_status_CE', 'employment_status_CF', 'employment_status_CG',
       'housing_status_BB', 'housing_status_BC', 'housing_status_BD',
       'housing_status_BE', 'housing_status_BF', 'housing_status_BG',
       'device_os_macintosh', 'device_os_other', 'device_os_windows',
       'device_os_x11']
    
    scaler = MinMaxScaler()
    
    train_scaled = pd.DataFrame(scaler.fit_transform(train[cols]), columns=cols, index=train.index)
    validate_scaled = pd.DataFrame(scaler.transform(validate[cols]), columns=cols, index=validate.index)
    test_scaled = pd.DataFrame(scaler.transform(test[cols]), columns=cols, index=test.index)
    
    return train_scaled, validate_scaled, test_scaled
#------------------------------------------------------------------------------
def prep_fraud_data(df):
    
    '''Preps the fraud data and returns the data split into train, validate and test portions'''
    
    # Convert binary categorical variables to numeric
    df['payment_type_encoded'] = df.payment_type.map({'AA': 0, 'AD': 1, 'AB': 2, 'AC': 3, 'AE': 4})
    df['employment_status_encoded'] = df.employment_status.map({'CB': 0, 'CA': 1, 'CC': 2, 'CF': 3, 'CD': 4, 'CE': 5, 'CG': 6})
    df['housing_status_encoded'] = df.housing_status.map({'BC': 0, 'BE': 1, 'BD': 2, 'BA': 3, 'BB': 4, 'BF': 5, 'BG': 6})
    df['device_os_encoded'] = df.device_os.map({'linux': 0, 'other': 1, 'windows': 2, 'x11': 3, 'macintosh': 4})
    df['source'] = df.source.map({'INTERNET': 0, 'TELEAPP': 1}) 
    
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['payment_type', \
                              'employment_status', \
                              'housing_status', \
                              'device_os', \
                              'source']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    df = df.drop(columns= ['payment_type', \
                              'employment_status', \
                              'housing_status', \
                              'device_os', \
                              'source']) 
    
    # split the data
    train, validate, test = split_data(df)
    
    return train, validate, test
#------------------------------------------------------------------------------

def acquire_fraud_df():
    '''
    This function will:
    '''
    # create the csv: 
    if os.path.isfile('base.csv'):  
        #if the csv file exists it will read the file
        df = pd.read_csv('base.csv', index_col = 0)
    
    else: 
        # create a table for red wine
        df = pd.read_csv('base.csv')
        df.to_csv('base.csv')
        
    return df
#------------------------------------------------------------------------------
#remove outliers function
def remove_outliers(df, k):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    cat_cols, num_cols = identify_columns(df)
    
    for col in num_cols:
        
        # For each column, it calculates the first quartile (q1) and 
        #third quartile (q3) using the .quantile() method, where q1 
        #corresponds to the 30th percentile and q3 corresponds to the 80th percentile.
        q1, q3 = df[col].quantile([.30, .80])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
    