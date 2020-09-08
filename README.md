# BIXI Case Study

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
>The mandate is to strengthen BIXI’s demand prediction capabilities. BIXI would like to estimate its bicycle demand by hour of the day based on their data from 2018.

## 2) Gather the Data
The data sets were provided. They are uploaded in the data sets folder.

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
The rides data set is separated by months and the geocoordinates of each station is in a separate CSV file. So I'll start by joining all of these files together so that all the variables can be accessed from a single dataset.<br>

Merging the months of the 2018 data set and outputting the data into a new file.<br>
<img src="/images/UNION_2018_BIXI_workflow.PNG" title="2018 BIXI rides" width="350" height="auto"/><br>

Then add the lattitude and longitude for each station (I left out the station name).<br>
<img src="/images/JOIN_2018_BIXI_Stations.PNG" title="2018 BIXI rides + geocoordinates" width="400" height="auto"/><br>

Next I joined the temperature to the BIXI and Stations combined data set into a new file called 2018_BIXI_Stations_Temperature.CSV. So now I have the list of all the rides, locations and temperature for the 2018 BIXI season. I left out the wind speed, humidity, etc.<br>
<img src="/images/JOIN_2018_BIXI_Stations_Temperature.PNG" title="2018 BIXI rides + geocoordinates + temp" width="600" height="auto"/><br>

### 3.5 Greet the data
**Import data**
```python
# read data set
BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Stations_Temperature.csv", encoding= 'unicode_escape')
```
**Preview data**
```python
# get a peek at the top 5 rows of the data set
print(BIXI_data.head())
```
```
   Month  Day  Hour  ... Wind Dir (10s deg)  Wind Spd (km/h) Stn Press (kPa)
0      4   11     0  ...                 20                7          101.14
1      4   11     0  ...                 20                7          101.14
2      4   11     0  ...                 20                7          101.14
3      4   11     0  ...                 20                7          101.14
4      4   11     0  ...                 20                7          101.14
```
**Date column types and count**
```python
# understand the type of each column
print(BIXI_data.info())
```
```
RangeIndex: 5223651 entries, 0 to 5223650
Data columns (total 17 columns):
 #   Column               Dtype  
---  ------               -----  
 0   Month                int64  
 1   Day                  int64  
 2   Hour                 int64  
 3   start_date           object 
 4   start_station_code   int64  
 5   end_date             object 
 6   end_station_code     int64  
 7   duration_sec         int64  
 8   is_member            int64  
 9   latitude             float64
 10  longitude            float64
 11  Temp (°C)            float64
 12  Dew Point Temp (°C)  float64
 13  Rel Hum (%)          int64  
 14  Wind Dir (10s deg)   int64  
 15  Wind Spd (km/h)      int64  
 16  Stn Press (kPa)      float64
dtypes: float64(5), int64(10), object(2)
```
**Summarize the central tendency, dispersion and shape**
```python
# get information on the numerical columns for the data set
with pd.option_context('display.max_columns', len(BIXI_data.columns)):
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

           is_member      latitude     longitude     Temp (°C)  \
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

        Dew Point Temp (°C)   Rel Hum (%)  Wind Dir (10s deg)  \
count               5223651       5223651             5223651   
unique                  NaN           NaN                 NaN   
top                     NaN           NaN                 NaN   
freq                    NaN           NaN                 NaN   
mean               9.845254      56.56023            18.82941   
std                  7.8752      18.91035            9.731781   
min                   -19.3            16                   0
25%                     4.3            41                  15   
50%                    10.7            56                  20   
75%                      16            71                  25   
max                    24.3            99                  36

        Wind Spd (km/h)  Stn Press (kPa)
count           5223651          5223651
unique              NaN              NaN
top                 NaN              NaN
freq                NaN              NaN
mean           6.283840         100.7311
std            2.828469        0.6325684
min                   1            98.39
25%                   4           100.34
50%                   6           100.76
75%                   8           101.13
max                  20           102.83
```

## 4) Data Cleaning
The data is cleaned in 2 steps:
1. Correcting outliers
2. Completing null or missing data

### 4.1 
From Tableau, I can see that only 10 out of 552 station codes are not within the 6000 to 7000 interval. So these 10 codes can be filtered out.<br>

### 4.2 Completing null or missing data
The columns containing null values need to be identified.<br>
**Training data**
```python
# find number of null values in each column
print('Number of null values per column:\n', BIXI_data.isnull().sum())
```
```
Number of null values per column:
Month                  0
Day                    0
Hour                   0
start_date             0
start_station_code     0
end_date               0
end_station_code       0
duration_sec           0
is_member              0
latitude               0
longitude              0
Temp (°C)              0
Dew Point Temp (°C)    0
Rel Hum (%)            0
Wind Dir (10s deg)     0
Wind Spd (km/h)        0
Stn Press (kPa)        0
dtype: int64
```

There aren't any null values for the data sets so there are no additional steps required at this point.<br>

## 5) Data Exploration
Let's look at the distribution for each column based on the number of rides.<br>
<img src="/images/Month_distribution.png" title="Distribution of BIXI rides by month" width="500" height="auto"/><br>
```python
print('Month:\n', BIXI_data.Month.value_counts(sort=False))
```
```
Month:
4     227516
5     805580
6     872410
7     944606
8     952142
9     796795
10    481450
11    143152
Name: Month, dtype: int64
```

<img src="/images/Day_distribution.png" title="Distribution of BIXI rides by day" width="500" height="auto"/><br>
```python
print('Day:\n', BIXI_data.Day.value_counts(sort=False))
```
```
Day:
1     177842
2     153570
3     156329
4     148174
5     178529
6     177106
7     180238
8     171886
9     189986
10    175735
11    174114
12    198274
13    184622
14    175525
15    168935
16    178420
17    167394
18    173067
19    170249
20    173739
21    167480
22    147810
23    174350
24    171623
25    143558
26    146963
27    176586
28    171761
29    170795
30    167036
31    111955
Name: Day, dtype: int64
```

<img src="/images/Hour_distribution.png" title="Distribution of BIXI rides by hour" width="500" height="auto"/><br>
```python
print('Hour:\n', BIXI_data.Hour.value_counts(sort=False))
```
```
Hour:
0      97050
1      61267
2      40778
3      33686
4      17933
5      18239
6      52950
7     165058
8     397512
9     290844
10    187416
11    209647
12    261441
13    274103
14    266581
15    300707
16    412008
17    547456
18    459474
19    340006
20    259309
21    215560
22    175223
23    139403
Name: Hour, dtype: int64
```
The overall Hours distribution shows two peaks: the first at around 8AM and the second at around 5PM. These peak times correspond exactly to when people go to work and when they get off work so there could be a correlation there. I'll confirm that by plotting the distribution of rides on weekdays vs. weekends.

<img src="/images/Hour_weekday_distribution.png" title="Distribution of BIXI rides by hour on weekdays" width="500" height="auto"/><br>

<img src="/images/Hour_weekend_distribution.png" title="Distribution of BIXI rides by hour on weekends" width="500" height="auto"/><br>

When comparing the weekday and weekend distribution, the graph clearly shows that the demand for BIXIs on weekdays correlates to the start and end of normal working hours. Whereas on weekends, the demand of BIXIs is high throughout the afternoon.

<img src="/images/Member_percent.png" title="BIXI membership percentage" width="500" height="auto"/><br>
```python
print('is_member:\n', BIXI_data.is_member.value_counts())
```
```
is_member:
1    4340146
0     883505
Name: is_member, dtype: int64
```

<img src="/images/Rides_by_membership.png" title="Rides by membership" width="500" height="auto"/><br>

The majority of riders are BIXI members. Based on the demand during weekdays, we can conclude that one of the reasons riders opted for a membership is to use BIXI to commute to work.

<img src="/images/Temperature_distribution.png" title="Distribution of BIXI rides by temperature" width="500" height="auto"/><br>
```python
print('Temp (°C):\n', BIXI_data.Temperature.value_counts())
```
```
Temp (°C):
 23.2    54492
 24.1    53673
 24.0    52818
 26.1    52734
 24.4    51002
         ...  
-5.9        34
-2.4        22
-5.8        22
-9.3        12
-9.4         8
Name: Temperature, Length: 411, dtype: int64
```

This graph shows that most rides took place when the temperature was above 0 degrees and lower than 30 degrees Celcius. This makes sense because riding a bike when it's freezing cold or extremely hot is not comfortable. The linear trend line returned a p-value of 0.0001 and a R-squared of 0.319993 which is a good indication that there is a correlation between the weather and BIXI demand.<br>

<img src="/images/Stations_distribution.png" title="Distribution of BIXI stations" width="500" height="auto"/><br>
```python
print('start_station_code:\n', BIXI_data.start_station_code.value_counts())
```
```
start_station_code:
6100    53768
6136    43562
6184    43342
6064    42344
6221    37053
        ...  
5005      749
5004      568
7009      551
5002      424
5003      339
Name: start_station_code, Length: 552, dtype: int64
```
This graph shows the number of BIXI rides by station. Some stations are more frequent than others. Note that there isn't a BIXI station for each value of the x-axis. For example, there isn't a BIXI station with a code of 4500, 4501, etc.<br>

<img src="/images/Duration_distribution.png" title="Distribution of duration of BIXI rides" width="500" height="auto"/><br>
```python
print('duration_sec:\n', BIXI_data.duration_sec.value_counts())
```
```
duration_sec:
342     5781
284     5755
289     5754
319     5751
338     5738
        ... 
6567       1
6023       1
6841       1
6268       1
6874       1
Name: duration_sec, Length: 7035, dtype: int64
```

**Trend Line Summary**
| Variable | Trend Line | R-Squared | p-value |
| :-------: | :---------:| :-------:| :-------:|
| Month | Polynomial | **0.968939** | 0.0017901 |
| Day (overall) | Polynomial | 0.315074 | 0.0154662 |
| Hour | Polynomial | **0.721809** | < 0.0001 |
| Weekday | Polynomial | 0.574977 | 0.0005573 |
| Weekend | Polynomial | **0.911839** | < 0.0001 |
| Temp (°C) | Polynomial | **0.71087** | < 0.0001 |
| Station Code | Polynomial | 0.166669 | < 0.0001 |
| Duration | Polynomial | **0.883909** | < 0.0001 |
| Rel Hum (%) | Polynomial | **0.857773** | < 0.0001 |
| Stn Press (kPa) | Polynomial | 0.575157 | < 0.0001 |
| Wind Dir (10s deg) | Polynomial | 0.14657 | 0.150621 |
| Wind Spd (km/h) | Polynomial | **0.899023** | < 0.0001 |

## 6) Feature Engineering
For this data set, I created a "ratio" feature which is calculated by dividing the number of bikes in by the number of bikes out for each station on a given day. This will determine which stations generally receive more bikes and which stations have more bikes departing from it.<br>

<img src="/images/Stations_Ratios_2018_BIXI.PNG" title="Ratios of Stations" width="auto" height="auto"/><br>

The Alteryx workflow will output the results into a file called 2018_BIXI_Stations_Temperature_Ratio.CSV.<br>

The temperature shows a clear correlation with demand but there are too many unique values. The temperature can be grouped into bins. I chose to split the temperatures in 10 bins of equal intervals.<br>

<img src="/images/Create_temperature_bins.PNG" title="Bins for Temperature" width="500" height="auto"/><br>

In addition, we saw that there is a higher correlation of the BIXI demand by hour on weekends.<br>

<img src="/images/Day_of_week_2018.PNG" title="Add Day of week feature" width="500" height="auto"/><br>

The objective is to predict the demand of BIXI stations but the target variable was not given as part of the initial dataset. The target variable needs to be defined as the amount of BIXI rides at a given station.<br>

<img src="/images/create_BIXI_count.PNG" title="BIXI demand" width="500" height="auto"/><br>

The final output file with all the features is called 2018_BIXI_Stations_Temperature_Ratio_DoW_Bins_Count.CSV.

### 6.1 Exploration of new features
The traffic of each BIXI station can vary depending on location. To find out which stations are the most popular (more bikes in than out), I plotted the map of BIXI stations and color coded the ratios.<br>

<img src="/images/Station_popularity.png" title="Distribution of BIXI rides by popularity" width="500" height="auto"/><br>

Downtown Montreal is a hotspot for riders to dock their bikes and stations closer to the river also receive more riders. On the other hand, the stations located out of downtown have more bikes out than in.<br>

<img src="/images/Temperature_bin_distribution.png" title="Distribution of BIXI rides by temperature bin" width="auto" height="600"/><br>

```python
# read data with new features created using Alteryx
new_BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature_Ratio_DoW_Bins_Count.csv", encoding= 'unicode_escape')

# explore amount of values per temperature bin
print('Temperature Bin:\n', new_BIXI_data.Temp_Bin.value_counts(sort=False))
```
```
Temperature Bin:
1       5845
2      10900
3      61658
4     122320
5     180826
6     224657
7     328382
8     387859
9     167277
10     22288
Name: Temp_Bin, dtype: int64
```

```python
# split into numerical values
df_numerical = new_BIXI_data[['Month', 'Day', 'Hour', 'is_Weekday', 'start_station_code', 
                              'Temp_Bin', 'Hum_Bin', 'Demand']]

# plot a heatmap showing the correlation between all numerical columns
print(df_numerical.corr())
```
```
                       Month       Day      Hour  ...  Temp_Bin   Hum_Bin    Demand
Month               1.000000 -0.164471 -0.007026  ... -0.214167  0.345659 -0.041753
Day                -0.164471  1.000000 -0.003996  ... -0.037382 -0.000179  0.002634
Hour               -0.007026 -0.003996  1.000000  ...  0.107208 -0.167751  0.130389
is_Weekday          0.012282  0.013649  0.034503  ...  0.005527  0.111213  0.038892
start_station_code  0.000239  0.001189  0.006504  ...  0.011690 -0.010254 -0.103149
Temp_Bin           -0.214167 -0.037382  0.107208  ...  1.000000 -0.220860  0.187901
Hum_Bin             0.345659 -0.000179 -0.167751  ... -0.220860  1.000000 -0.155117
Demand             -0.041753  0.002634  0.130389  ...  0.187901 -0.155117  1.000000
```
```python
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
        annot_kws={'fontsize':10 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(df_numerical)
plt.show()
```
<img src="/images/num_heatmap.png" title="Correlation between numerical columns" width="700" height="auto"/><br>

- "Hour" and "Temperature" shows the highest correlation in regards to the "Demand".
- The "Humidity" and "Month" seem to indicate some correlation.

### 6.2 Split into Training and Testing Data
Finally, the data set was split into a training(80%) and testing(20%) set using Alteryx.<br>

<img src="/images/SPLIT_TRAIN_TEST.PNG" title="Split Train Test" width="400" height="auto"/><br>
```python
# read train data
train_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Train_Data.csv", encoding= 'unicode_escape')

# read test data
test_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/2018_BIXI_Test_Data.csv", encoding= 'unicode_escape')

# create a copy of train data to start exploring/modifying it
train_copy = train_data.copy(deep = True)

print("All Data Shape: {}".format(new_BIXI_data.shape))
print("Train Data Shape: {}".format(train_data.shape))
print("Test Data Shape: {}".format(test_data.shape))
```
```
All Data Shape: (1512012, 8)
Train Data Shape: (1209610, 8)
Test Data Shape: (302402, 8)
```

## 7) Evaluate Model Performance

### 7.1 Data Preprocessing for Model
```python
# define target variable y of the training data set
y_train = train_copy["Demand"]

# define features to be used for the predictive models
features = ['Month', 'Hour', 'DayofWeek', 'start_station_code', 'Temp_Bin']

# define x-axis variables for training and testing data sets
x_train = pd.get_dummies(train_copy[features])
x_test = pd.get_dummies(test_data[features])
```

### 7.2 Model Building

```python
# Gaussian Naive Bayes
gnb = GaussianNB()
cv = cross_val_score(gnb, x_train, y_train, cv=5)
print(cv)
print(cv.mean())
```
```
[0.02744725 0.03144114 0.03100562 0.03154882 0.02617183]
0.029522932373137033
```

## 8) Model Tuning

## 9) Validate Data Model

## 10) Conclusion

