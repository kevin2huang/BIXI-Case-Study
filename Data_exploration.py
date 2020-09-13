import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read train data set
BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature.csv", encoding= 'unicode_escape')

# explore the amount of unique variables
BIXI_data.columns = ['Month', 'Day', 'Hour', 'start_date', 'start_station_code', 'end_date',
                     'end_station_code', 'duration_sec', 'is_member', 'latitude',
                     'longitude', 'Temperature', 'Dew_point', 'Humidity',
                     'Wind_dir', 'Wind_spd', 'Stn_pressure']

# print('Month:\n', BIXI_data.Month.value_counts(sort=False))
# print('Day:\n', BIXI_data.Day.value_counts(sort=False))
# print('Hour:\n', BIXI_data.Hour.value_counts(sort=False))
# print('start_station_code:\n', BIXI_data.start_station_code.value_counts())
# print('end_station_code:\n', BIXI_data.end_station_code.value_counts())
# print('duration_sec:\n', BIXI_data.duration_sec.value_counts())
# print('is_member:\n', BIXI_data.is_member.value_counts())
# print('Temp (°C):\n', BIXI_data.Temperature.value_counts())
# print('Dew Point Temp (°C):\n', BIXI_data.Dew_point.value_counts())
# print('Rel Hum (%):\n', BIXI_data.Humidity.value_counts())
# print('Wind Dir (10s deg):\n', BIXI_data.Wind_dir.value_counts())
# print('Wind Spd (km/h):\n', BIXI_data.Wind_spd.value_counts())
# print('Stn Press (kPa):\n', BIXI_data.Stn_pressure.value_counts())


print('Month skewness:\n', BIXI_data.Month.skew())
print('Day skewness:\n', BIXI_data.Day.skew())
print('Hour skewness:\n', BIXI_data.Hour.skew())
print('duration_sec skewness:\n', BIXI_data.duration_sec.skew())
print('Temp (°C) skewness:\n', BIXI_data.Temperature.skew())
print('Dew Point Temp (°C) skewness:\n', BIXI_data.Dew_point.skew())
print('Rel Hum (%) skewness:\n', BIXI_data.Humidity.skew())
print('Wind Dir (10s deg) skewness:\n', BIXI_data.Wind_dir.skew())
print('Wind Spd (km/h) skewness:\n', BIXI_data.Wind_spd.skew())
print('Stn Press (kPa) skewness:\n', BIXI_data.Stn_pressure.skew())

# read data with new features created using Alteryx
all_BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature_Ratio_DoW_Bins_Demand.csv", encoding= 'unicode_escape')

# split into numerical values
df_numerical = all_BIXI_data[['Month', 'Day', 'Hour', 'is_Weekend', 'Duration_Bin', 'Temp_Bin', 'Hum_Bin', 'Wind_spd', 'Demand']]

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
        annot_kws={'fontsize':11 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

# correlation_heatmap(df_numerical)
# plt.show()