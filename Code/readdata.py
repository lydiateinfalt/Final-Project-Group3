import pandas as pd

# Crashes reported in DC
# Reference: https://opendata.dc.gov/datasets/crash-details-table
crash = pd.read_csv("Crash_Details_Table.csv")
crash = pd.DataFrame(crash, columns=['CRIMEID', 'PERSONID', 'PERSONTYPE', 'AGE', 'FATAL', 'MAJORINJURY', 'MINORINJURY', 'VEHICLEID', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING'])


# Analysis of each data set
print("Total number of rows in data set: ", crash.shape[0])
print("Total number of columns in data set:", crash.shape[1])

# Drop any duplicate rows based on CRIMEID
crash.drop_duplicates(subset=['CRIMEID'])
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
