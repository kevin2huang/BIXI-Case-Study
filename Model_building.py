import numpy as np 
import pandas as pd
import sklearn
import itertools
import copy
import csv
import openpyxl

#Common Model Algorithms
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import StandardScaler
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
		
		Feature.append(new_target)
		Mean.append(cv.mean())

		print(new_target)
		print("Mean: {}".format(cv.mean()))
		print("-"*10)
		
		combinations(new_target, new_data, model)

scale = StandardScaler()

# Feature = []
# Mean = []
# print("Gaussian Naive Bayes")
# combinations([], features, gnb)
# Combination = [Feature, Mean]
# df = pd.DataFrame(Combination)
# df.to_excel('GNB_comb.xlsx', header=False, index=False)

# define features to be used for the predictive models
# print(train_copy.columns)
features = [ 'Month', 'Day', 'Hour', 'duration_log', 'Wind_spd',
             'Hum_Bin', 'Stn_pressure', 'Temp_Bin', 'Wind_dir' ]


# define x-axis variables for training and testing data sets
x_train = pd.get_dummies(train_copy[features])
x_train_scaled = scale.fit_transform(x_train)

x_test = pd.get_dummies(test_data[features])
x_test_scaled = scale.fit_transform(x_test)

# define target variable y
y_train = train_copy.Demand
y_test = test_data.Demand

# Gaussian Naive Bayes
print("Gaussian Naive Bayes")
gnb = GaussianNB()
cv = cross_val_score(gnb, x_train_scaled, y_train, cv=10, scoring='accuracy')
print(cv)
print(cv.mean())

# Linear Regression
print("Linear Regression")
lin_r = LinearRegression()
cv = cross_val_score(lin_r, x_train_scaled, y_train, cv=5)
print(cv)
print(cv.mean())

# Logistic Regression
print("Logistic Regression")
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr, x_train_scaled, y_train, cv=5)
print(cv)
print(cv.mean())

# Decision Tree
print("Decision Tree")
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt, x_train_scaled, y_train, cv=5)
print(cv)
print(cv.mean())

# k-Neighbors
print("k-Neighbors")
knn = KNeighborsClassifier()
cv = cross_val_score(knn, x_train_scaled, y_train, cv=5)
print(cv)
print(cv.mean())

# Random Forest
print("Random Forest")
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf, x_train_scaled, y_train, cv=5)
print(cv)
print(cv.mean())

# SVC
print("SVC")
svc = SVC(probability = True)
cv = cross_val_score(svc, x_train_scaled, y_train, cv=5)
print(cv)
print(cv.mean())

# XGB
print("XGB")
xgb = XGBClassifier(random_state = 1)
cv = cross_val_score(xgb, x_train_scaled, y_train, cv=5)
print(cv)
print(cv.mean())

# group of models
estimator = [('lr', lr),
	         ('knn', knn),
	         ('rf', rf),
	         ('gnb', gnb),
	         ('svc', svc),
	         ('xgb', xgb)]

# Voting Classifier with soft voting
# print("Voting Classifier")
vot_soft = VotingClassifier(estimators = estimator, voting = 'soft') 
cv = cross_val_score(vot_soft, x_train_scaled, y_train, cv=5, scoring='accuracy')
print(cv)
print(cv.mean())

vot_soft.fit(x_train_scaled, y_train)
y_predict = vot_soft.predict(x_test_scaled)

print("MSE: {}".format(mean_absolute_error(y_test, y_predict)))

# 0.09071144763475524
# [Finished in 465.8s]

# using accuracy_score metric to predict accuracy
# score = accuracy_score(y_test, y_predict) 
# print("Soft Voting Score % d" % score)


# submission = pd.DataFrame({ 'start_station_code' : test_data.start_station_code, 
#                			  'Month' : test_data.Month, 
#                			  'Hour' : test_data.Hour, 
#                			  'is_Weekend' : test_data.is_Weekend, 
#                			  'Temp_Bin' : test_data.Temp_Bin, 
#                			  'Hum_Bin' : test_data.Hum_Bin, 
#                			  'duration_sec' : test_data.duration_sec, 
#                			  'Wind_spd' : test_data.Wind_spd,
#                			  'Demand' : test_data.Demand,
#                			  'Prediction' : y_predict })

# submission.to_csv('predictions.csv', index=False)


"""

8) Model Tuning

"""

