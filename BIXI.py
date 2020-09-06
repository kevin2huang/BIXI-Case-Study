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
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
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
train_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Stations_Temperature_Train.csv", encoding= 'unicode_escape')

# read test data set
test_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Stations_Temperature_Test.csv", encoding= 'unicode_escape')

# create a copy of train data to start exploring/modifying it
train_copy = train_data.copy(deep = True)

# get a peek at the top 5 rows of the training data
# print(train_copy.head())

# understand the type of each column
# print(train_copy.info())

# get information on the numerical columns for the training data set
# with pd.option_context('display.max_columns', 12):
    # print(train_copy.describe(include='all'))


"""

4.2) Completing null or missing data

"""

# find number of null values in each column
# print('Number of null values per column for train data:\n', train_copy.isnull().sum())

# find number of null values in each column
# print('Number of null values per column for test data:\n', test_data.isnull().sum())

"""

5) Data Exploration

"""

# explore the amount of unique variables
train_copy.columns = ['Month', 'Day', 'Hour', 'start_date', 'start_station_code', 'end_date',
       'end_station_code', 'duration_sec', 'is_member', 'latitude',
       'longitude', 'Temperature']

print('Month:\n', train_copy.Month.value_counts(sort=False))
print('Day:\n', train_copy.Day.value_counts(sort=False))
print('Hour:\n', train_copy.Hour.value_counts(sort=False))
print('start_station_cod:\n', train_copy.start_station_code.value_counts())
print('end_station_code:\n', train_copy.end_station_code.value_counts())
print('duration_sec:\n', train_copy.duration_sec.value_counts())
print('is_member:\n', train_copy.is_member.value_counts())
print('Temp (°C):\n', train_copy.Temperature.value_counts())