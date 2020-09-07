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
# sns.heatmap(df_numerical.corr())
# plt.show()


"""

6.1) Exploration of new features

"""
# read data with new features created using Alteryx
new_BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature_Ratio_DoW_Bins_Count.csv", encoding= 'unicode_escape')

# explore amount of values per temperature bin
print('Temperature Bin:\n', new_BIXI_data.Temp_Bin.value_counts(sort=False))




"""

6.3) Split into Training and Testing Data

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