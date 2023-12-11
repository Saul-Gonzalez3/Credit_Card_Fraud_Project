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

import sklearn.linear_model

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


def evaluate_reg(y, yhat):
        '''
        based on two series, y_act, y_pred, (y, yhat), we
        evaluate and return the root mean squared error
        as well as the explained variance for the data.
        
        returns: rmse (float), rmse (float)
        '''
        rmse = mean_squared_error(y, yhat, squared=False)
        r2 = r2_score(y, yhat)
        return rmse, r2
#------------------------------------------------------------------------------

def get_baseline(train):
    X = train.drop(columns={'fraud_bool'})
    y = train['fraud_bool']
    
    from sklearn.linear_model import LinearRegression
    baseline = y.mean()
    y.shape
    
    baseline_array = np.repeat(baseline, y.shape[0])
    baseline_rmse, baseline_r2 = evaluate_reg(y, baseline_array)

    eval_df = pd.DataFrame([{
        'model': 'baseline',
        'rmse': baseline_rmse,
        'r2': baseline_r2
    }])
    print(f"""
_______________________________________________________________
Baseline: {baseline} | Baseline RMSE: {baseline_rmse} | Baseline r2: {baseline_r2}""")
    return baseline, baseline_rmse, baseline_r2

#------------------------------------------------------------------------------
def OLS(X_train, y_train, baseline, X_val, y_val):
    from sklearn.linear_model import LinearRegression
    # MAKE THE THING: create the model object
    linear_model = LinearRegression()
    #1. FIT THE THING: fit the model to training data
    OLSmodel = linear_model.fit(X_train, y_train)

    #2. USE THE THING: make a prediction
    y_train_pred = linear_model.predict(X_train)
    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train, y_train_pred) ** (.5) # 0.5 to get the root
    
    # convert results into dataframe
    result = pd.DataFrame({
        "target": y_train,
        "OLS_prediction": y_train_pred,
        "baseline_pred": baseline
    })
    
    # convert to dataframe
    X_val = pd.DataFrame(X_val)
    X_val[X_val.columns] = X_val
    X_val = X_val[X_val.columns]
    
    #2. USE THE THING: make a prediction
    y_val_pred = linear_model.predict(X_val)
    
    #3. Evaluate: RMSE
    rmse_val = mean_squared_error(y_val, y_val_pred) ** (.5) # 0.5 to get the root
    val_rmse, val_r2 = evaluate_reg(y_val, linear_model.predict(X_val))
    OLSmodel.coef_
    
    print(f"""RMSE for Ordinary Least Squares
_____________________
Training/In-Sample: {rmse_train}, 
Validation/Out-of-Sample:  {rmse_val}
Difference:  {rmse_val - rmse_train}
Difference from baseline:  {rmse_val - baseline}
Val r2: {val_r2}""")

#------------------------------------------------------------------------------
def LassoLars(X_train, y_train, baseline, X_val, y_val):
    from sklearn.linear_model import LassoLars

    # MAKE THE THING: create the model object
    lars = LassoLars(alpha= 1.0)
    
    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    laslars = lars.fit(X_train, y_train)
    
    #2. USE THE THING: make a prediction
    y_train_pred_lars = lars.predict(X_train)
    
    #3. Evaluate: RMSE
    rmse_train_lars = mean_squared_error(y_train, y_train_pred_lars) ** (0.5)
    
    # predict validate
    y_val_pred_lars = lars.predict(X_val)
    
    # evaluate: RMSE
    rmse_val_lars = mean_squared_error(y_val, y_val_pred_lars) ** (0.5)
    val_rmse, val_r2 = evaluate_reg(y_val, lars.predict(X_val))
    # how important is each feature to the target
    laslars.coef_
    
    print(f"""RMSE for Lasso + Lars
_____________________
Training/In-Sample: {rmse_train_lars}, 
Validation/Out-of-Sample:  {rmse_val_lars}
Difference:  {rmse_val_lars - rmse_train_lars}
Difference from baseline:  {rmse_val_lars - baseline}
Val r2: {val_r2}""")
#------------------------------------------------------------------------------

def GLM(X_train, y_train, baseline, X_val, y_val):
    from sklearn.linear_model import TweedieRegressor
    # MAKE THE THING: create the model object
    glm_model = TweedieRegressor(alpha= 1.0, power= 1)
    
    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    tweedieReg = glm_model.fit(X_train, y_train)
    
    #2. USE THE THING: make a prediction
    y_train_pred_tweedie = glm_model.predict(X_train)
    
    #3. Evaluate: RMSE
    rmse_train_tweedie = mean_squared_error(y_train, y_train_pred_tweedie) ** (0.5)
    
    # predict validate
    y_val_pred_tweedie = glm_model.predict(X_val)
    
    # evaluate: RMSE
    rmse_val_tweedie = mean_squared_error(y_val, y_val_pred_tweedie) ** (0.5)
    
    val_rmse, val_r2 = evaluate_reg(y_val, glm_model.predict(X_val))
    # how important is each feature to the target
    tweedieReg.coef_
    
    print(f"""RMSE for GLM
_____________________
Training/In-Sample: {rmse_train_tweedie}, 
Validation/Out-of-Sample:  {rmse_val_tweedie}
Difference:  {rmse_val_tweedie - rmse_train_tweedie}
Difference from baseline:  {rmse_val_tweedie - baseline}
Val r2: {val_r2}""")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def break_em_out(preprocessed_train, preprocessed_validate, preprocessed_test):
    # train set
    X_train = preprocessed_train.drop(columns=['fraud_bool']) 
    y_train = preprocessed_train['fraud_bool']
    
    # validate set
    X_val = preprocessed_validate.drop(columns=['fraud_bool']) 
    y_val = preprocessed_validate['fraud_bool']
    
    # test
    X_test = preprocessed_test.drop(columns=['fraud_bool'])
    y_test = preprocessed_test['fraud_bool']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

#------------------------------------------------------------------------------
def OLS_test(X_train, y_train, baseline, X_val, y_val, X_test, y_test):
    from sklearn.linear_model import LinearRegression
    # MAKE THE THING: create the model object
    linear_model = LinearRegression()
    #1. FIT THE THING: fit the model to training data
    OLSmodel = linear_model.fit(X_train, y_train)

    #2. USE THE THING: make a prediction
    y_train_pred = linear_model.predict(X_train)
    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train, y_train_pred) ** (.5) # 0.5 to get the root
    
    # convert results into dataframe
    result = pd.DataFrame({
        "target": y_train,
        "OLS_prediction": y_train_pred,
        "baseline_pred": baseline
    })
    
        # convert to dataframe
    X_val = pd.DataFrame(X_val)
    X_val[X_val.columns] = X_val
    X_val = X_val[X_train.columns]
    
    #2. USE THE THING: make a prediction
    y_val_pred = linear_model.predict(X_val)
    
    #3. Evaluate: RMSE
    rmse_val = mean_squared_error(y_val, y_val_pred) ** (.5) # 0.5 to get the root
    
    # convert to dataframe
    X_test = pd.DataFrame(X_test)
    X_test[X_test.columns] = X_test
    X_test = X_test[X_train.columns]
    
    #2. USE THE THING: make a prediction
    y_test_pred = linear_model.predict(X_test)
    
    #3. Evaluate: RMSE
    rmse_test = mean_squared_error(y_test, y_test_pred) ** (.5) # 0.5 to get the root
    test_rmse, test_r2 = evaluate_reg(y_test, linear_model.predict(X_test))   

    OLSmodel.coef_

    
    print(f"""RMSE & r2 for Ordinary Least Squares Test Model
_____________________________________________
Baseline: {baseline}
Training/In-Sample: {rmse_train} 
Validation/Out-of-Sample:  {rmse_val}
Test/Out-of-Sample:  {rmse_test}
Difference from baseline:  {rmse_test - baseline}
Test r2: {test_r2}""")
    
#------------------------------------------------------------------------------
def poly_model(X_train, y_train, baseline, X_val, y_val):
    
    from sklearn.preprocessing import PolynomialFeatures

    #Make the thing
    pf = PolynomialFeatures(degree=2)
    
    #fit the thing and USE it on train
    x_train_sq = pf.fit_transform(X_train)
    
    #Use the thing on Val 
    x_val_sq = pf.transform(X_val)
    
    # Make a new model for our polynomial regressor:
    
    plyreg = LinearRegression()
    plyreg.fit(x_train_sq, y_train)
        
    #Evaluate
    
    rmse, r_2 = evaluate_reg(y_train, plyreg.predict(x_train_sq))  
    val_rmse, val_r2 = evaluate_reg(y_val, plyreg.predict(x_val_sq))   
    print(f"""RMSE for Polynomial Regression
    _____________________
    Training/In-Sample: {rmse}, 
    Validation/Out-of-Sample:  {val_rmse}
    Difference:  {val_rmse - rmse}
    Difference from baseline:  {val_rmse - baseline}
    Val r2: {val_r2}""")
    
    #------------------------------------------------------------------------------
def poly_model_test(X_train, y_train, baseline, X_val, y_val, X_test, y_test):
    from sklearn.preprocessing import PolynomialFeatures

    #Make the thing
    pf = PolynomialFeatures(degree=4)
    
    #fit the thing and USE it on train
    x_train_sq = pf.fit_transform(X_train)
    
    #Use the thing on Val and Test
    x_val_sq = pf.transform(X_val)
    x_test_sq = pf.transform(X_test)
    # Make a new model for our polynomial regressor:
    
    plyreg = LinearRegression()
    plyreg.fit(x_train_sq, y_train)
        
    #Evaluate
    
    rmse, r_2 = evaluate_reg(y_train, plyreg.predict(x_train_sq))  
    val_rmse, val_r2 = evaluate_reg(y_val, plyreg.predict(x_val_sq))  
    test_rmse, test_r2 = evaluate_reg(y_test, plyreg.predict(x_test_sq))
    print(f"""RMSE & r2 for Polynomial Regression Test Model
    _____________________
    Baseline: {baseline}
    Training/In-Sample: {rmse}, 
    Validation/Out-of-Sample:  {val_rmse}
    Test/Out-of-Sample:  {test_rmse}
    Difference:  {val_rmse - rmse}
    Difference from baseline:  {val_rmse - baseline}
    Test r2: {test_r2}""")

def logit(train, X_train, y_train, validate, X_val, y_val,baseline_accuracy):
    # Create the logistic regression
    logit = LogisticRegression(random_state=123)
    
    # specify the features we're using
    features = ['customer_age', 'credit_risk_score', 'proposed_credit_limit','device_os_encoded', 'device_os_windows']
    
    # Fit a model using only these specified features
    # logit.fit(X_train[["age", "pclass", "fare"]], y_train)
    logit.fit(X_train[features], y_train)
    
    # Since we .fit on a subset, we .predict on that same subset of features
    y_pred = logit.predict(X_train[features])
    y_pred1 = logit.predict(X_val[features])
    print("Baseline is", round(baseline_accuracy, 2))
    print("Logistic Regression using Customer Age, Credit Risk Score, Proposed Credit Limit, Deviced OS Encoded and Device OS Windows")
    print('Accuracy of Logistic Regression classifier on training set: {:.4f}'
         .format(logit.score(X_train[features], y_train)))
    print('Logit1 model using Customer Age, Credit Risk Score, Proposed Credit Limit, Deviced OS Encoded and Device OS Windows')
    print(classification_report(y_val, y_pred1))
    
def logit1(train, X_train, y_train,validate, X_val, y_val, baseline_accuracy):
    # Create the logistic regression
    logit1 = LogisticRegression(random_state=123)
    
    # specify the features we're using
    features = ['income', 'name_email_similarity',
           'prev_address_months_count', 'current_address_months_count',
           'customer_age','date_of_birth_distinct_emails_4w', 
           'credit_risk_score', 'proposed_credit_limit', 
           'keep_alive_session',
           'device_distinct_emails_8w']
    
    # Fit a model using only these specified features
    logit1.fit(X_train[features], y_train)
    
    y_pred = logit1.predict(X_train[features])
    y_pred1 = logit1.predict(X_val[features])
    print("Logistic Regression using the Top 10 features")
    print('Accuracy of Logistic Regression classifier on training set: {:.4f}'
         .format(logit1.score(X_train[features], y_train)))
    print('Logit1 model using using the Top 10 features')
    print(classification_report(y_val, y_pred1))
    
def logit2(train, X_train, y_train,validate, X_val, y_val, baseline_accuracy):
    # All features, all default hyperparameters
    logit2 = LogisticRegression(random_state=123)
    
    logit2.fit(X_train, y_train)
    
    y_pred = logit2.predict(X_train)
    y_pred1 = logit2.predict(X_val)
    print("Model trained on all features")
    print('Accuracy of Logistic Regression classifier on training set: {:.4f}'
         .format(logit2.score(X_train, y_train)))
    print("Logit2 model using all features and all model defaults")
    print(classification_report(y_val, y_pred1))
    
    
    
def eval_logit_on_Test(train, X_train, y_train, validate, X_val, y_val, test, X_test, y_test, baseline_accuracy):
    # Create the logistic regression
    logit = LogisticRegression(random_state=123)
    
    # Let's determine logit's metrics on validate
    features = ['customer_age', 'credit_risk_score', 'proposed_credit_limit','device_os_encoded', 'device_os_windows']
    
    logit.fit(X_train[features], y_train)
    
    y_pred = logit.predict(X_train[features])
    y_pred1 = logit.predict(X_val[features])
    y_pred2 = logit.predict(X_test[features])
    
    print('Logit1 model using Customer Age, Credit Risk Score, Proposed Credit Limit, Deviced OS Encoded and Device OS Windows')
    print(classification_report(y_test, y_pred2))