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

### 3.3 Data dictionary
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

**Montreal 2018 Temperature** (The climate records come from the Government of Canada website. To simplify the analysis, I will only be using the weather data from the McTavish reservoir station as a proxy for all the weather patterns of the different areas of the island of Montreal.)<br>
| Variable | Definition | Key |
| :-------: | :---------:| :-------:|
| Date/Time | The date and time |  |
| Year | The year extracted from the Date/Time column |  |
| Month | The month extracted from the Date/Time column |  |
| Day | The day extracted from the Date/Time column |  |
| Time | The time extracted from the Date/Time column |  |
| Temp (°C) | The temperature in celcius |  |
| Dew Point Temp (°C) | The dew point in celcius |  |
| Rel Hum (%) | The percent of relative humidity |  |
| Wind Dir (10s deg) | The wind direction by 10s of degrees |  |
| Wind Spd (km/h) | The speed of the wind in km/h |  |
| Stn Press (kPa) | The standard pressure in kPa |  |

### 3.4 Data set restructuring
The rides data set is separated by months and the geocoordinates of each station is in a separate CSV file. So I'll start by joining all of these files together so that all the variables can be accessed. To do this, I'll use Alteryx.<br>

I started by merging the months of the 2018 data set and outputting the data into a new file called OD_2018_all_months.CSV.<br>
<img src="/images/UNION_2018_BIXI_workflow.PNG" title="2018 BIXI rides" width="400" height="auto"/><br>
Then add the lattitude and longitude for each station (I left out the station name).<br>
<img src="/images/JOIN_2018_BIXI_Stations.PNG" title="2018 BIXI rides + geocoordinates" width="400" height="auto"/><br>
Next I joined the temperature to the BIXI and Stations combined data set into a new file called 2018_BIXI_Stations_Temperature.CSV. So now I have the list of all the rides, locations and temperature for the 2018 BIXI season. I left out the wind speed, humidity, etc.<br>
<img src="/images/JOIN_2018_BIXI_Stations_Temperature.PNG" title="2018 BIXI rides + geocoordinates + temp" width="600" height="auto"/><br>
Finally, the data set was split into a training(80%) and testing(20%) set.
<img src="/images/SPLIT_TRAIN_TEST.PNG" title="Split Train Test" width="400" height="auto"/><br>

### 3.5 Greet the data
**Import data**
```python
# read train data set
train_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Stations_Temperature_Train.csv", encoding= 'unicode_escape')

# create a copy of train data to start exploring/modifying it
train_copy = train_data.copy(deep = True)
```
**Preview data**
```python
# get a peek at the top 5 rows of the training data
print(train_copy.head())
```
```
   Month  Day  Hour  ...   latitude  longitude Temp (°C)
0      4   11     0  ...  45.511673 -73.562042       0.6
1      4   11     0  ...  45.518890 -73.563530       0.6
2      4   11     0  ...  45.502054 -73.573465       0.6
3      4   11     0  ...  45.507402 -73.578444       0.6
4      4   11     0  ...  45.507402 -73.578444       0.6
```
**Date column types and count**
```python
# understand the type of each column
print(train_copy.info())
```
```
RangeIndex: 4178921 entries, 0 to 4178920
Data columns (total 12 columns):
 #   Column              Dtype  
---  ------              -----  
 0   Month               int64  
 1   Day                 int64  
 2   Hour                int64  
 3   start_date          object 
 4   start_station_code  int64  
 5   end_date            object 
 6   end_station_code    int64  
 7   duration_sec        int64  
 8   is_member           int64  
 9   latitude            float64
 10  longitude           float64
 11  Temp (°C)           float64
dtypes: float64(3), int64(7), object(2)
```
**Summarize the central tendency, dispersion and shape**
```python
# get information on the numerical columns for the data set
with pd.option_context('display.max_columns', 12):
    print(train_copy.describe(include='all'))
```
```
           Month           Day          Hour        start_date  \
count    4178921       4178921       4178921           4178921
unique       NaN           NaN           NaN            287273
top          NaN           NaN           NaN  2018-06-12 17:09
freq         NaN           NaN           NaN                86
mean    7.267364      15.72647      14.20463               NaN
std     1.778885      8.767025      5.300635               NaN
min            4             1             0               NaN
25%            6             8            10               NaN
50%            7            16            15               NaN
75%            9            23            18               NaN
max           11            31            23               NaN

       start_station_code           end_date  end_station_code  duration_sec \
count             4178921            4178921           4178921       4178921
unique                 NaN            286693               NaN           NaN
top                    NaN  2018-05-29 17:43               NaN           NaN
freq                   NaN                82               NaN           NaN
mean             6331.976                NaN          6327.396      800.6714
std              415.2819                NaN          429.9209      605.8301
min                  4000                NaN              4000            61
25%                  6114                NaN              6100           369
50%                  6211                NaN              6205           643
75%                  6397                NaN              6405          1075
max                 10002                NaN             10002          7199

           is_member      latitude     longitude     Temp (°C)
count                      4178921       4178921       4178921
unique           NaN           NaN           NaN           NaN
top              NaN           NaN           NaN           NaN
freq             NaN           NaN           NaN           NaN
mean       0.8308130      45.51737     -73.57979      19.58934
std        0.3749170    0.02118086    0.02083352      7.398439
min                0      45.42947     -73.66739         -10.7
25%                1      45.50373     -73.58976          15.1
50%                1      45.51941     -73.57635          21.2
75%                1      45.53167     -73.56545            25
max                1      45.58276     -73.49507          35.8
```

## 4) Data Cleaning
The data is cleaned in 4 steps:
1. Correcting outliers
2. Completing null or missing data
3. Creating new features
4. Converting/Formatting datatypes

### 4.1 Correcting outliers
Based on the summary above, there aren't any obvious outliers so I skipped this for now. Some outliers may be identified during the data exploration.

### 4.2 Completing null or missing data
The columns containing null values need to be identified for both the training and test data sets.<br>
**Training data**
```python
# find number of null values in each column
print('Number of null values per column for train data:\n', train_copy.isnull().sum())
```
```
Number of null values per column for train data:
Month                 0
Day                   0
Hour                  0
start_date            0
start_station_code    0
end_date              0
end_station_code      0
duration_sec          0
is_member             0
latitude              0
longitude             0
Temp (°C)             0
dtype: int64
```
**Testing data**
```python
# find number of null values in each column
print('Number of null values per column for test data:\n', test_data.isnull().sum())
```
```
Number of null values per column for test data:
Month                 0
Day                   0
Hour                  0
start_date            0
start_station_code    0
end_date              0
end_station_code      0
duration_sec          0
is_member             0
latitude              0
longitude             0
Temp (°C)             0
dtype: int64
```
There aren't any null values for either train or test sets.

## 5) Data Exploration
Let's look at the distribution for each column based on the number of rides.<br>
<img src="/images/Month_distribution.png" title="Distribution of BIXI rides by month" width="500" height="auto"/><br>
```
Month:
4     182183
5     643942
6     698231
7     755511
8     762220
9     637312
10    384976
11    114546
Name: Month, dtype: int64
```

<img src="/images/Day_distribution.png" title="Distribution of BIXI rides by day" width="500" height="auto"/><br>
```
Day:
1     141975
2     122977
3     125215
4     118463
5     142648
6     141348
7     144381
8     137525
9     151748
10    140580
11    139363
12    158823
13    147832
14    140710
15    134940
16    142692
17    133639
18    138495
19    136057
20    138889
21    134152
22    118169
23    139634
24    137261
25    114937
26    117510
27    141612
28    137538
29    136726
30    133541
31     89541
Name: Day, dtype: int64
```

<img src="/images/Hour_distribution.png" title="Distribution of BIXI rides by day" width="500" height="auto"/><br>
```
Hour:
0      77585
1      49081
2      32696
3      26876
4      14405
5      14620
6      42359
7     132110
8     318224
9     232269
10    149916
11    167944
12    209297
13    219174
14    213518
15    240697
16    329268
17    437753
18    367625
19    272174
20    207645
21    172181
22    139929
23    111575
Name: Hour, dtype: int64
```

## 6) Model Building

## 7) Model Tuning

## 8) Validate Data Model

## 9) Conclusion

