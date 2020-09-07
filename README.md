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
6) [Feature Engineering](#6-feature-engineering)<br>
7) [Model Building](#7-model-building)<br>
8) [Model Tuning](#8-model-tuning)<br>
9) [Validate Data Model](#9-validate-data-model)<br>
10) [Conclusion](#10-conclusion)

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
The rides data set is separated by months and the geocoordinates of each station is in a separate CSV file. So I'll start by joining all of these files together so that all the variables can be accessed from a single dataset. To do this, I'll use Alteryx.<br>

I started by merging the months of the 2018 data set and outputting the data into a new file called OD_2018_all_months.CSV.<br>
<img src="/images/UNION_2018_BIXI_workflow.PNG" title="2018 BIXI rides" width="400" height="auto"/><br>
Then add the lattitude and longitude for each station (I left out the station name).<br>
<img src="/images/JOIN_2018_BIXI_Stations.PNG" title="2018 BIXI rides + geocoordinates" width="400" height="auto"/><br>
Next I joined the temperature to the BIXI and Stations combined data set into a new file called 2018_BIXI_Stations_Temperature.CSV. So now I have the list of all the rides, locations and temperature for the 2018 BIXI season. I left out the wind speed, humidity, etc.<br>
<img src="/images/JOIN_2018_BIXI_Stations_Temperature.PNG" title="2018 BIXI rides + geocoordinates + temp" width="600" height="auto"/><br>

### 3.5 Greet the data
**Import data**
```python
# read train data set
BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Stations_Temperature.csv", encoding= 'unicode_escape')
```
**Preview data**
```python
# get a peek at the top 5 rows of the training data
print(BIXI_data.head())
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
print(BIXI_data.info())
```
```
RangeIndex: 5223651 entries, 0 to 5223650
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
    print(BIXI_data.describe(include='all'))
```
```
           Month           Day          Hour        start_date  \
count    5223651       5223651       5223651           5223651
unique       NaN           NaN           NaN            292467
top          NaN           NaN           NaN  2018-08-14 17:08
freq         NaN           NaN           NaN               106
mean    7.267356      15.72403      14.20556               NaN
std     1.778988      8.767144      5.300635               NaN
min            4             1             0               NaN
25%            6             8            10               NaN
50%            7            16            15               NaN
75%            9            23            18               NaN
max           11            31            23               NaN

       start_station_code           end_date  end_station_code  duration_sec \
count             5223651            5223651           5223651       5223651
unique                 NaN            291710               NaN           NaN
top                    NaN  2018-09-06 17:39               NaN           NaN
freq                   NaN               101               NaN           NaN
mean             6331.992                NaN          6327.396      800.7508
std              415.4750                NaN          430.1704      606.0940
min                  4000                NaN              4000            61
25%                  6114                NaN              6100           370
50%                  6211                NaN              6205           643
75%                  6397                NaN              6405          1075
max                 10002                NaN             10002          7199

           is_member      latitude     longitude     Temp (°C)
count        5223651       5223651       5223651       5223651
unique           NaN           NaN           NaN           NaN
top              NaN           NaN           NaN           NaN
freq             NaN           NaN           NaN           NaN
mean       0.8308645      45.51737     -73.57979      19.58934
std        0.3748716    0.02118324    0.02083564      7.398439
min                0      45.42947     -73.66739         -10.7
25%                1      45.50373     -73.58976          15.2
50%                1      45.51941     -73.57635          21.2
75%                1      45.53167     -73.56545            25
max                1      45.58276     -73.49507          35.8
```

## 4) Data Cleaning
The data is cleaned in 2 steps:
1. Correcting outliers
2. Completing null or missing data

### 4.1 Correcting outliers
Based on the summary above, there aren't any obvious outliers so I skipped this for now. Some outliers may be identified during the data exploration.

### 4.2 Completing null or missing data
The columns containing null values need to be identified for both the training and test data sets.<br>
**Training data**
```python
# find number of null values in each column
print('Number of null values per column:\n', BIXI_data.isnull().sum())
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

There aren't any null values for the data sets so there are no additional steps required at this point.<br>



## 5) Data Exploration
Let's look at the distribution for each column based on the number of rides.<br>
<img src="/images/Month_distribution.png" title="Distribution of BIXI rides by month" width="500" height="auto"/><br>
```python
print('Month:\n', train_copy.Month.value_counts(sort=False))
```
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
```python
print('Day:\n', train_copy.Day.value_counts(sort=False))
```
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

<img src="/images/Hour_distribution.png" title="Distribution of BIXI rides by hour" width="500" height="auto"/><br>
```python
print('Hour:\n', train_copy.Hour.value_counts(sort=False))
```
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
The overall Hours distribution shows two peaks: the first at around 8AM and the second at around 5PM. These peak times correspond exactly to when people go to work and when they get off work so there could be a correlation there. I'll confirm that by plotting the distribution of rides on weekdays vs. weekends.

<img src="/images/Hour_weekday_distribution.png" title="Distribution of BIXI rides by hour on weekdays" width="500" height="auto"/><br>

<img src="/images/Hour_weekend_distribution.png" title="Distribution of BIXI rides by hour on weekends" width="500" height="auto"/><br>

When comparing the weekday and weekend distribution, the graph clearly shows that the demand for BIXIs on weekdays correlates to the start and end of normal working hours. Whereas on weekends, the demand of BIXIs is high throughout the afternoon.

<img src="/images/Member_percent.png" title="BIXI membership percentage" width="500" height="auto"/><br>

<img src="/images/Rides_by_membership.png" title="Rides by membership" width="500" height="auto"/><br>

The majority of riders are BIXI members. Based on the demand during weekdays, we can conclude that one of the reasons riders opted for a membership is to use BIXI to commute to work.

<img src="/images/Temperature_distribution.png" title="Distribution of BIXI rides by temperature" width="500" height="auto"/><br>

This graph shows that most rides took place when the temperature was above 0 degrees and lower than 30 degrees Celcius. This makes sense because riding a bike when it's freezing cold or extremely hot is not comfortable. The linear trend line returned a p-value of 0.0001 and a R-squared of 0.319993 which is a good indication that there is a correlation between the weather and BIXI demand.<br>

<img src="/images/Stations_distribution.png" title="Distribution of BIXI stations" width="500" height="auto"/><br>

This graph shows the number of BIXI rides by station. Some stations are more frequent than others. Note that there isn't a BIXI station for each value of the x-axis. For example, there isn't a BIXI station with a code of 4500, 4501, etc. 

```python
# split into numerical values
df_numerical = BIXI_data[['is_member', 'Month', 'Day', 'Hour', 'start_station_code', 
							'end_station_code', 'duration_sec', 'latitude', 'longitude', 
							'Temperature']]

# plot a heatmap showing the correlation between all numerical columns
print(df_numerical.corr())
sns.heatmap(df_numerical.corr())
plt.show()
```
<img src="/images/num_heatmap.png" title="Correlation between numerical columns" width="600" height="auto"/><br>
```
                    is_member     Month  ...  longitude  Temperature
is_member            1.000000  0.033615  ...  -0.066868    -0.095171
Month                0.033615  1.000000  ...  -0.005889    -0.193358
Day                  0.000508 -0.155403  ...  -0.002202    -0.033552
Hour                -0.036288 -0.017236  ...   0.016287     0.141312
start_station_code   0.024376 -0.000506  ...  -0.197350    -0.011260
end_station_code     0.020379 -0.001493  ...  -0.086674    -0.002884
duration_sec        -0.274457 -0.057599  ...   0.014850     0.094586
latitude             0.063988  0.007384  ...  -0.127652    -0.029114
longitude           -0.066868 -0.005889  ...   1.000000     0.018355
Temperature         -0.095171 -0.193358  ...   0.018355     1.000000
```

## 6) Feature Engineering
For this data set I created a "ratio" feature which is calculated by dividing the number of bikes in by the number of bikes out for each station on a given day. This will determine which stations generally receive more bikes and which stations have more bikes leaving it.

<img src="/images/Stations_Ratios_2018_BIXI.PNG" title="Ratios of Stations" width="auto" height="auto"/><br>

The Alteryx workflow will output the results into a file called 2018_BIXI_Stations_Temperature_Ratio.CSV.<br>

Since the objective is to predict the demand of BIXI stations, the target variable would need to be defined. In this case, the target variable would be the amount of BIXI rides at a given station. I'll create an Alteryx workflow to create that column.<br>

### 6.1 Exploration of new features
The traffic of each BIXI station can vary depending on location. To find out which stations are the most popular (more bikes in than out), I plotted the map of BIXI stations and color coded the ratios.<br>

This outputs the results into a file titled 2018_BIXI_Stations_Temperature_Ratio_Train.CSV which is the same Training data set plus the ratio column.<br>

<img src="/images/Station_popularity.png" title="Distribution of BIXI rides by temperature" width="500" height="auto"/><br>

Downtown Montreal is a hotspot for riders to dock their bikes and stations closer to the river also receive more riders. On the other hand, the stations located out of downtown have more bikes out than in. 


### 6.2 Convert Formats


### 6.3 Split into Training and Testing Data
Finally, the data set was split into a training(80%) and testing(20%) set using Alteryx.<br>

<img src="/images/SPLIT_TRAIN_TEST.PNG" title="Split Train Test" width="400" height="auto"/><br>
```python
# read train data
train_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Train_Data.csv", encoding= 'unicode_escape')

# read test data
test_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Test_Data.csv", encoding= 'unicode_escape')

# create a copy of train data to start exploring/modifying it
train_copy = train_data.copy(deep = True)

print("All Data Shape: {}".format(BIXI_data.shape))
print("Train Data Shape: {}".format(train_data.shape))
print("Test Data Shape: {}".format(test_data.shape))
```
```
All Data Shape: (5223651, 12)
Train Data Shape: (4178921, 13)
Test Data Shape: (1044730, 13)
```

## 6) Model Building

## 7) Model Tuning

## 8) Validate Data Model

## 9) Conclusion

