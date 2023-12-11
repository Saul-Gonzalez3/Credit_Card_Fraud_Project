# <a name="top"></a>Life, Liberty, and the Pursuit of Freedom from Credit Card Fraud

by Saul Gonzalez


> *The challenge for capitalism is that the things that breed trust also breed the environment for fraud.*
> > James Surowiecki
 

[Project Plan](#Project_Plan) | [Data Dictionary](#Data_Dictionary) | [Conclusions](#Conclusions) | [Next Steps](#Next_Steps) | [Recommendations](#Recommendations) | [Steps to Reproduce Our Work](#Steps_to_Reproduce_My_Work)|

***
<h3><b>Project Description:</b></h3>  

This project contains the findings of research derived from the utilization of linear and logistic regression machine learning models paired with feature selection to determine the highest drivers, or indicators, that predict credit card fraud. The data obtained for this research was acquired from Kaggle's Bank Account Fraud (BAF) suite of datasets containing dynamic, biased, and unbalanced synthetic data primed to offer a realistic present-day real-world dataset for fraud detection. The outcome of this project should help propel future efforts using machine learning and feature selection to mitigate credit card fraud.
    
***
<h3><b>Project Goal:</b></h3>  Predict credit card fraud while incorporating unsupervised machine learning learning techniques that can consistently and accurately detect fraudulent transactions on unseen data.

***
<h4><b>Initial Questions:</b></h4>

1. Are any of the features correlated? 

2. Does <b>device_os</b> show a liability for credit card fraud due to the percentage of fraud attributed to they type of device_os?

3. Does <b>zip_count_4w</b> show a trend of credit card fraud attributed to specific locations?

4. Classification or regression? Should I do both for a comparison given the time I have to work on this?

5. Are all input variables relevant? Which ones are <b>MOST</b> relevant? 

***
<a name="Project_Plan"></a><h3><b>Project Plan:</b></h3>

1. Create all the files needed to make a functioning project (.py and .ipynb files).

2. Create a .gitignore file and ignore the env.py file.

3. Start by acquiring data from [Kaggle Bank Account Fraud Dataset Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) dataset (hit the download button for 'Base.csv') and document all my initial acquisition steps in the acquire.py file.

4. Using the prepare file, clean the data and split it into train, validatate, and test sets.

5. Explore the data. (Focus on the main main questions). Experiment with various feature combinations and clustering to gain insight, if some is found, to support hypothesis testing.

6. Answer all the questions with statistical testing.

7. Identify drivers of credit card fraud. Make prediction of credit card fraud using driving features found during exploration and statistical testing.

8. Document findings.

9. Add important finding to the final notebook.

10. Create csv file of test predictions on best performing model.

[[Back to top](#top)]

***
<a name="Data_Dictionary"></a><h3><b>Data Dictionary:</b></h3>

|**Input Variables**|**Description**|
|----------|----------------|
|Fixed fraud_bool| Fraud label (1 if fraud, 0 if legit).|
|Income | Annual income of the applicant in quantiles. Ranges between [0, 1]. |
| name_email_similarity | Metric of similarity between email and applicant’s name. Higher values represent higher similarity. Ranges between [0, 1].|
| prev_address_months_count | Number of months in previous registered address of the applicant, i.e. the applicant’s previous residence, if applicable. Ranges between [−1, 380] months (-1 is a missing value). |
| current_address_months_count | Months in currently registered address of the applicant. Ranges between [−1, 406] months (-1 is a missing value).|
| customer_age | Applicant’s age in bins per decade (e.g, 20-29 is represented as 20). |
| days_since_request | Number of days passed since application was done. Ranges between [0, 78] days. |
| intended_balcon_amount | Initial transferred amount for application. Ranges between [−1, 108]. |
| payment_type | Credit payment plan type. 5 possible (annonymized) values. |
| zip_count_4w | Number of applications within same zip code in last 4 weeks. Ranges between [1, 5767]. |

[[Back to top](#top)]


***
<a name="Conclusions"></a><h3><b>Conclusions:</b></h3>

After acquiring & preparing the data, I conducted uni/bi/multi-variate exploration on the credit card fraud data to look at features and how they might impact the target fraud_bool.

I used a heatmap to look at all possible correlations and ranked correlations to the target. I made 5 pairs of the top 10 features and conducted statistical testing to observe potential relationships between the features.

The results of my data exploration culminated in me using a diverse combination of features in the modeling phase of this project.

The Select K Best Top 5 Features: Customer_age, Credit_risk_score, Proposed_credit_limit, Device_os_encoded, Device_os_windows.

I originally chose to go with linear regression modeling due to the large number of my features being continuous.

I used the following linear regression models:

- Ordinary Least Squares (OLS) 
- LassoLars 
- Generalized Linear Model (GLM) 
- Polynomial Regression

I found that our Ordinary Least Squares model was the 'best' performing model, ultimately performing 9% worse than the baseline.

Upon further research, I then decided to continue further analysis by utilizing logistic regression modeling because my target was binary and that this method would likely provide better results if I looked at 'precision' regarding building a model for predicting fraud.

I conducted three logistic regression models using my ranked features (Top 5 for one, Top 10 for the second, and All features for the third). I conducted the tests on train and validate data and found no difference in any of the results when looking for precision.
    
[[Back to top](#top)]
    

***    
<a name="Next_Steps"></a><h3><b>Next Steps:</b></h3>

-The nexts steps would be to look at conducting this entire study from different angles. Incorporating a technique like anomaly detection might better serve me in developing something that better attacks the cause of detecting credit card fraud.

- Furthermore, additional study on logistic regression would likely provide me the ability to dig deeper into this research and achieve better results by understanding and using my tools at the mastery level.
    
[[Back to top](#top)]
    

***    
<a name="Recommendations"></a><h3><b>Recommendations:</b></h3>  

- The data source is extremely unbalanced in regards to the target (1.16% of data is fraudulent, which is what I want to predict). A rebalance should be conducted prior to further study using this data source for better results.

- I would consult other data science studies on credit card fraud to identify if there are commonly discussed features that are not found in this study. It would be important to identify potential blindspots that would improve the precision of the selected model, should the feature be available in our data, but not considered for some reason.

- Learning more advanced machine learning techniques could be beneficial to producing a better model that predicts credit card fraud with precision.
    
[[Back to top](#top)]
    

***    
<a name="Steps_to_Reproduce_My_Work"></a><h3><b>Steps to Reproduce My Work:</b></h3>

1. Clone this repo.

2. Acquire the data from the Kaggle Bank Account Fraud Dataset Suite Dataset.

3. Put the data in the file containing the cloned repo.

4. Run your notebook.
    
[[Back to top](#top)]

