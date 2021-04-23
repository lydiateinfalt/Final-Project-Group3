import pandas as pd
import numpy as np

# Crashes reported in DC
# Reference: https://opendata.dc.gov/datasets/crash-details-table
crash = pd.read_csv('https://opendata.arcgis.com/datasets/70248b73c20f46b0a5ee895fc91d6222_25.csv')
#crash = pd.read_csv("Crash_Details_Table.csv")
# Analysis of each data set
print("Original data set")
print("Total number of rows in data set: ", crash.shape[0])
print("Total number of columns in data set:", crash.shape[1])

crash = pd.DataFrame(crash, columns=['PERSONID', 'PERSONTYPE', 'AGE', 'FATAL', 'MAJORINJURY', 'MINORINJURY', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING'])
print("Selecting columns from data set")
print("Total number of rows in data set: ", crash.shape[0])
print("Total number of columns in data set:", crash.shape[1])

crash['FATALMAJORINJURIES'] = np.where((crash['FATAL'].eq('Y') | crash['MAJORINJURY'].eq('Y')),1,0)
print("Adding new column FATALMAJORINJURIES")
print("Total number of rows in data set: ", crash.shape[0])
print("Total number of columns in data set:", crash.shape[1])
fatal_crash=crash[crash.FATALMAJORINJURIES.eq(1.0)]
print("Total number observations with FATALMAJORINJURES", fatal_crash.shape[0])
print("Percentage of FATAL crashes", (fatal_crash.shape[0]/crash.shape[0]*100))


print("Dataset first few rows:\n ")
print(crash.head())

print ('-'*80 + '\n')

# printing the structure of the dataset
print("Dataset info:\n ")
print(crash.info())
print ('-'*80 + '\n')
# printing the summary statistics of the dataset
print(crash.describe(include='all'))
print ('-'*80 + '\n')
print("Fatal crashes")
print(crash[crash['FATAL'] == 'Y'])
crash.to_csv("crash.csv")