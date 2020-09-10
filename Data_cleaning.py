import numpy as np 
import pandas as pd

# read train data set
BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature.csv", encoding= 'unicode_escape')

# find number of null values in each column
print('Number of null values per column:\n', BIXI_data.isnull().sum())