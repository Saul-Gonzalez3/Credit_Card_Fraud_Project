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



def eval_Spearman(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a monotonic relationship.
Spearman’s r: {r:2f}
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a monotonic relationship.
Spearman’s r: {r:2f}
P-value: {p}""")


    
def eval_Pearson(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a linear relationship with a Correlation Coefficient of {r:2f}.
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a linear relationship.
Pearson’s r: {r:2f}
P-value: {p}""")

    


def eval_dist(r, p, α=0.05):
    if p > α:
        return print(f"""The data is normally distributed""")
    else:
        return print(f"""The data is NOT normally distributed""")

    
def heatmap_ranked(train):
    from matplotlib import patches
    df = train
    target='fraud_bool'

    plt.figure(figsize=(5,10))
    ax = sns.heatmap(train[abs(train.corr()[target]).sort_values(ascending=False).index].corr()[target].to_frame()[1:],
                        annot=True, cmap='Purples', vmin=-1, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(right=False, labelsize=8) 
    cbar.set_ticks([-1, -.5, 0, .5, 1])
    plt.tick_params(axis='both', left=False, bottom=False)

    rectangle = patches.Rectangle((0, 0), 1, 10, linewidth=1.5, edgecolor='#C40000', facecolor='none')
    ax.add_patch(rectangle)

    plt.title('Top 10 Correlated Features of Fraud')
    plt.show()