import pandas as pd

# read train data set
BIXI_data = pd.read_csv("Data sets/Bixi Montreal Rentals 2018/Output from Alteryx/2018_BIXI_Stations_Temperature.csv", encoding= 'unicode_escape')

# get a peek at the top 5 rows of the data set
print(BIXI_data.head())

# understand the type of each column
print(BIXI_data.info())

# get information on the numerical columns for the data set
with pd.option_context('display.max_columns', len(BIXI_data.columns)):
    print(BIXI_data.describe(include='all'))