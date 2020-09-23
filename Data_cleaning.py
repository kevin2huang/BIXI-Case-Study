import numpy as np 
import pandas as pd

# read train data set
BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature.csv", encoding= 'unicode_escape')

# columns
BIXI_data.columns = ['Month', 'Day', 'Hour', 'start_date', 'start_station_code', 'end_date',
                     'end_station_code', 'duration_sec', 'is_member', 'latitude',
                     'longitude', 'Temperature', 'Dew_point', 'Humidity',
                     'Wind_dir', 'Wind_spd', 'Stn_pressure']

# find number of null values in each column
print('Number of null values per column:\n', BIXI_data.isnull().sum())

# find which columns need to be normalized
print('Month skewness: ', BIXI_data.Month.skew())
print('Day skewness: ', BIXI_data.Day.skew())
print('Hour skewness: ', BIXI_data.Hour.skew())
print('duration_sec skewness: ', BIXI_data.duration_sec.skew())
print('Temp (°C) skewness: ', BIXI_data.Temperature.skew())
print('Dew Point skewness: ', BIXI_data.Dew_point.skew())
print('Rel Hum (%) skewness: ', BIXI_data.Humidity.skew())
print('Wind Dir (10s deg) skewness: ', BIXI_data.Wind_dir.skew())
print('Wind Spd (km/h) skewness: ', BIXI_data.Wind_spd.skew())
print('Stn Press (kPa) skewness: ', BIXI_data.Stn_pressure.skew())