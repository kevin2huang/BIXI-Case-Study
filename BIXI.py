"""

3.1, 3.2) Import Libraries

"""

import sys
import numpy as np 
import pandas as pd
import scipy as sp
import IPython
from IPython import display
import sklearn
import random
import time

# print("Python version: {}". format(sys.version))
# print("pandas version: {}". format(pd.__version__))
# print("matplotlib version: {}". format(matplotlib.__version__))
# print("seaborn version: {}". format(sns.__version__))
# print("NumPy version: {}". format(np.__version__))
# print("SciPy version: {}". format(sp.__version__)) 
# print("IPython version: {}". format(IPython.__version__)) 
# print("scikit-learn version: {}". format(sklearn.__version__))

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

#Common Model Algorithms
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics


"""

3.5) Greet the Data

"""

# read train data set
BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature.csv", encoding= 'unicode_escape')

# get a peek at the top 5 rows of the training data
# print(BIXI_data.head())

# understand the type of each column
# print(BIXI_data.info())

# get information on the numerical columns for the training data set
# with pd.option_context('display.max_columns', 12):
    # print(BIXI_data.describe(include='all'))


"""

4.2) Completing null or missing data

"""

# find number of null values in each column
# print('Number of null values per column for train data:\n', BIXI_data.isnull().sum())


"""

5) Data Exploration

"""

# explore the amount of unique variables
BIXI_data.columns = ['Month', 'Day', 'Hour', 'start_date', 'start_station_code', 'end_date', 
                      'end_station_code', 'duration_sec', 'is_member', 'latitude', 'longitude', 
                      'Temperature']

# print('Month:\n', train_copy.Month.value_counts(sort=False))
# print('Day:\n', train_copy.Day.value_counts(sort=False))
# print('Hour:\n', train_copy.Hour.value_counts(sort=False))
# print('start_station_cod:\n', train_copy.start_station_code.value_counts())
# print('end_station_code:\n', train_copy.end_station_code.value_counts())
# print('duration_sec:\n', train_copy.duration_sec.value_counts())
# print('is_member:\n', train_copy.is_member.value_counts())
# print('Temp (°C):\n', train_copy.Temperature.value_counts())

# split into numerical values
df_numerical = BIXI_data[['is_member', 'Month', 'Day', 'Hour', 'start_station_code', 
							'end_station_code', 'duration_sec', 'latitude', 'longitude', 
							'Temperature']]

# plot a heatmap showing the correlation between all numerical columns
# print(df_numerical.corr())

#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

# correlation_heatmap(df_numerical)
# plt.show()


"""

6.1) Exploration of new features

"""
# read data with new features created using Alteryx
new_BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature_Ratio_DoW_Bins_Count.csv", encoding= 'unicode_escape')

# explore amount of values per temperature bin
# print('Temperature Bin:\n', new_BIXI_data.Temp_Bin.value_counts(sort=False))


"""

6.2) Split into Training and Testing Data

"""

# read train data
train_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Train_Data.csv", encoding= 'unicode_escape')

# read test data
test_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Test_Data.csv", encoding= 'unicode_escape')

# create a copy of train data to start exploring/modifying it
train_copy = train_data.copy(deep = True)

# print("All Data Shape: {}".format(BIXI_data.shape))
# print("Train Data Shape: {}".format(train_data.shape))
# print("Test Data Shape: {}".format(test_data.shape))


"""

7.1) Data Preprocessing for Model

"""

# define target variable y of the training data set
y_train = train_copy["Demand"]

# define features to be used for the predictive models
features = ['Month', 'Hour', 'DayofWeek', 'start_station_code', 'Temp_Bin']

# define x-axis variables for training and testing data sets
x_train = pd.get_dummies(train_copy[features])
x_test = pd.get_dummies(test_data[features])


"""

7.2) Model Building

"""

# Gaussian Naive Bayes
gnb = GaussianNB()
cv = cross_val_score(gnb, x_train, y_train, cv=5)
print(cv)
print(cv.mean())

