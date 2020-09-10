import sys
import numpy as np 
import pandas as pd
import scipy as sp
import sklearn
import random
import time
import itertools
import copy

#Common Model Algorithms
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_absolute_error


# read train data
train_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Train_Data.csv", encoding= 'unicode_escape')

# read test data
test_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Test_Data.csv", encoding= 'unicode_escape')

# create a copy of train data to start exploring/modifying it
train_copy = train_data.copy(deep = True)

# print("All Data Shape: {}".format(new_BIXI_data.shape))
# print("Train Data Shape: {}".format(train_data.shape))
# print("Test Data Shape: {}".format(test_data.shape))

# helper method to use every combination of features and run it through the model
def combinations(target, data, model):
	for i in range(len(data)):
		new_target = copy.copy(target)
		new_data = copy.copy(data)
		new_target.append(data[i])
		new_data = data[i+1:]
		cv = cross_val_score(model, pd.get_dummies(train_copy[new_target]), y_train, cv=10)
		print(new_target)
		print(cv)
		print(cv.mean())
		print("-"*10)
		
		combinations(new_target, new_data)

scale = StandardScaler()

# define features to be used for the predictive models
features = ['start_station_code', 'Month', 'Day', 'Hour', 'is_Weekend',
       'duration_sec', 'Temp_Bin', 'Hum_Bin', 'Stn_pressure', 'Wind_dir',
       'Wind_spd', 'Avg_Ratio', 'is_member']

# define x-axis variables for training and testing data sets
x_train = pd.get_dummies(train_copy[features])
# x_train_scaled = scale.fit_transform(train_dummies)

x_test = pd.get_dummies(test_data[features])
# x_test_scaled = scale.fit_transform(test_dummies)

# define target variable y
y_train = train_copy.Demand
y_test = test_data.Demand

# Gaussian Naive Bayes
gnb = GaussianNB()
# cv = cross_val_score(gnb, x_train_scaled, y_train, cv=10)
# print(cv)
# print(cv.mean())

# [0.07496252 0.06921539 0.06146927 0.06421789 0.05647176 0.06471764
#  0.05972014 0.06671664 0.07048238 0.06923269]
# 0.06572063256050056

# Logistic Regression
lr = LogisticRegression(max_iter = 20000)
# cv = cross_val_score(lr, x_train_scaled, y_train, cv=5)
# print(cv)
# print(cv.mean())

# [0.03798101 0.05509745 0.0535982  0.0496064  0.0378608 ]
# 0.04682877229384808

# Decision Tree
dt = tree.DecisionTreeClassifier(random_state = 1)
# cv = cross_val_score(dt, x_train_scaled, y_train, cv=5)
# print(cv)
# print(cv.mean())

# [0.13843078 0.19677661 0.19965017 0.19080345 0.14457079]
# 0.17404636117527889

# k-Neighbors
knn = KNeighborsClassifier()
# cv = cross_val_score(knn, x_train_scaled, y_train, cv=5)
# print(cv)
# print(cv.mean())

# [0.07396302 0.08783108 0.0892054  0.08684243 0.06310134]
# 0.08018865426714358

# Random Forest
rf = RandomForestClassifier(random_state = 1)
# cv = cross_val_score(rf, x_train_scaled, y_train, cv=5)
# print(cv)
# print(cv.mean())

# [0.11831584 0.18990505 0.18815592 0.17580907 0.13419968]
# 0.1612771116628366

# SVC
svc = SVC(probability = True)
# cv = cross_val_score(svc, x_train_scaled, y_train, cv=5)
# print(cv)
# print(cv.mean())

# [0.06921539 0.08795602 0.0655922  0.06135199 0.05772835]
# 0.06836879261231561

# XGB
xgb = XGBClassifier(random_state = 1)
# cv = cross_val_score(xgb, x_train_scaled, y_train, cv=5)
# print(cv)
# print(cv.mean())


# group of models
estimator = [('lr', lr),
	         ('knn',knn),
	         ('rf',rf),
	         ('gnb',gnb),
	         ('svc',svc),
	         ('xgb',xgb)]

# Voting Classifier with hard voting
# vot_hard = VotingClassifier(estimators = estimator, voting ='hard')
# vot_hard.fit(x_train_scaled, y_train)
# y_predict = vot_hard.predict(x_test_scaled)

# using accuracy_score metric to predict accuracy 
# score = accuracy_score(y_test, y_predict) 
# print("Hard Voting Score % d" % score) 

# Voting Classifier with soft voting
vot_soft = VotingClassifier(estimators = estimator, voting = 'soft') 
vot_soft.fit(x_train, y_train)
y_predict = vot_soft.predict(x_test)

print(mean_absolute_error(y_test, y_predict))

# using accuracy_score metric to predict accuracy
# score = accuracy_score(y_test, y_predict) 
# print("Soft Voting Score % d" % score)


# submission = pd.DataFrame({ 'start_station_code' : test_data.start_station_code, 
#                				'Month' : test_data.Month, 
#                				'Hour' : test_data.Hour, 
#                				'is_Weekend' : test_data.is_Weekend, 
#                				'Temp_Bin' : test_data.Temp_Bin, 
#                				'Hum_Bin' : test_data.Hum_Bin, 
#                				'duration_sec' : test_data.duration_sec, 
#                				'Wind_spd' : test_data.Wind_spd,
#                				'Demand' : test_data.Demand,
#                				'Prediction' : y_predict })

# submission.to_csv('predictions.csv', index=False)


"""

8) Model Tuning

"""

