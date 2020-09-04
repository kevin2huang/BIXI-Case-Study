# BIXI Case Study

This case study was part of my Data Science for Business Decisions course at McGill.

## Technology and Resources Used

**Python Version**: 3.7.7<br>
**Tools**: Tableau, Alteryx

## Table of Contents
1) [Define the Problem](#1-define-the-problem)<br>
2) [Gather the Data](#2-gather-the-data)
3) [Prepare Data for Consumption](#3-prepare-data-for-consumption)<br>
4) [Data Cleaning](#4-data-cleaning)<br>
5) [Data Exploration](#5-data-exploration)<br>
6) [Model Building](#6-model-building)<br>
7) [Model Tuning](#7-model-tuning)<br>
8) [Validate Data Model](#8-validate-data-model)<br>
9) [Conclusion](#9-conclusion)

## 1) Define the Problem
The case study was provided.
>The mandate is to strengthen BIXI’s demand prediction capabilities. BIXI would like to estimate its bicycle demand based on their data from 2018.

## 2) Gather the Data
The data sets were provided by the course. I uploaded them in the data sets folder.

## 3) Prepare Data for Consumption

### 3.1 Import Libraries
The following code is written in Python 3.7.7. Below is the list of libraries used.
```python
import sys
import numpy as np 
import pandas as pd
import matplotlib
import seaborn as sns
import scipy as sp
import IPython
from IPython import display
import sklearn
import random
import time
```
My library versions at the time were:
```
Python version: 3.7.7
pandas version: 1.0.5
matplotlib version: 3.3.0
seaborn version: 0.10.1
NumPy version: 1.19.1
SciPy version: 1.5.2
IPython version: 7.17.0
scikit-learn version: 0.23.2
```
### 3.2 Load Data Modeling Libraries
These are the most common machine learning and data visualization libraries.
```python
#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
```

### 3.3 Get to know the data
The data dictionary for the data sets are as follows:<br>
**BIXI rides** (title in format OD_yyyy-mm)<br>
| Variable | Definition | Key |
| :-------: | :---------:| :-------:|
| start_date | The time and date a BIXI ride started |  |
| start_station_code | The ID of the station where the BIXI originated from |  |
| end_date | The time and date a BIXI ride started |  |
| end_station_code | The ID of the station where the BIXI arrived at |  |
| duration_sec | The duration in seconds of the ride |  |
| is_member | If the rider holds a BIXI membership or not | 1 = Yes, 0 = No  |

**BIXI Stations** (titled in the format Stations_yyyy)<br>
| Variable | Definition | Key |
| :-------: | :---------:| :-------:|
| code | The unique ID of the station |  |
| name | The name/intersection of the station |  |
| latitude | The latitude of the station |  |
| longitude | The longitude of the station |  |

**Montreal Temperature** (The climate records come from the Government of Canada website. To simplify the analysis, I will only be using the weather data from the McTavish reservoir station as a proxy for all the weather patterns of the different areas of the island of Montreal.)<br>
| Variable | Definition | Key |
| :-------: | :---------:| :-------:|
| Date/Time | The date and time |  |
| Year | The year extracted from Date/Time column |  |
| Month | The month extracted from Date/Time column |  |
| Day | The day extracted from Date/Time column |  |
| Time | The time extracted from Date/Time column |  |
| Temp (°C) | The temperature in celcius |  |
| Dew Point Temp (°C) | The dew point in celcius |  |
| Rel Hum (%) | The percent of relative humidity |  |
| Wind Dir (10s deg) | The wind direction by 10s of degrees |  |
| Wind Spd (km/h) | The speed of the wind in km/h |  |
| Stn Press (kPa) | The standard pressure in kPa |  |

## 4) Data Cleaning
The data is cleaned in 4 steps:
1. Correcting outliers
2. Completing null or missing data
3. Creating new features
4. Converting/Formatting datatypes

### 4.1 Correcting outliers

### 4.2 Completing null or missing data

## 5) Data Exploration

## 6) Model Building

## 7) Model Tuning

## 8) Validate Data Model

## 9) Conclusion

