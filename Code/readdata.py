import pandas as pd
import numpy as np
import seaborn as sns

# All crashes reported in DC
# Reference: https://opendata.dc.gov/datasets/crashes-in-dc?geometry=13.108%2C-75.996%2C-10.798%2C89.827
crashDC = pd.read_csv("Crashes_in_DC.csv")
columns1 = crashDC.dtypes
# Detailed crash report join with above table using CRIMEID
# Reference: https://opendata.dc.gov/datasets/crash-details-table
crashDCdets = pd.read_csv("Crash_Details_Table.csv")
columns2 = crashDCdets.dtypes
# Join two data sets and create a new df
crashAll = pd.merge(crashDC, crashDCdets,"inner", on="CRIMEID")

print()

# Display column names and data type
columns = crashAll.dtypes
print(crashAll.dtypes)
print(crashAll.head(10))
print(crashAll.tail(10))
print("Total number of rows in data set: ", crashAll.size)


crashAll['YEAR'] = crashAll['REPORTDATE'].str[0:4]
crashAll['MONTH'] = crashAll['REPORTDATE'].str[5:7]
crashAll['DAY'] = crashAll['REPORTDATE'].str[8:10]
print()

# fatalcrash= crashAll['FATAL' == 'Y']
fatalbikes = crashAll['FATAL_BICYCLIST' == 1]
fataldriver = crashAll['FATAL_DRIVER' == 1]
fatalpedestrian = crashAll['FATAL_PEDESTRIAN' == 1]


sns.countplot()