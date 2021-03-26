# Group 3 Preprocessing

# Arianna Code Start:
import pandas as pd
import data_cleanup
# import other libraries below:

# This is my directory on my personal computer so I've commented out and added other lines below for
# group members to paste their own directory.
# in_dc = pd.read_csv('/Users/ariannadunham/Desktop/Data Mining DATS 6103/Project/Crashes_in_DC.csv')
# details = pd.read_csv('/Users/ariannadunham/Desktop/Data Mining DATS 6103/Project/Crash_Details_Table.csv')

# Uncomment and paste directory below
# in_dc = pd.read_csv()
# details = pd.read_csv()

crash = data_cleanup.data
print(crash.shape)
# Dropping duplicate entries in CRIMEID(the unique identifier for this set) column of the main dataset (in_dc)
#in_dc = data.drop_duplicates(subset=['CRIMEID'])

# Checking to see the number of crashes involving someone 100+ to determine if that's a good
# Age to cap at
print("Crashes involving someone who is 100 or older:")
print(crash[crash.AGE >= 100.0].count()) # Result is 271

# Dropping all ages below 0 and above 100
age_filter = (crash.AGE > 100.0) | (crash.AGE < 0.0)
crash = crash.where(~age_filter)

# Dropping state abbreviations that do not exist (Ot, ou, vi, pu, un)
plate_filter = (crash.LICENSEPLATESTATE == 'OT') | (crash.LICENSEPLATESTATE == 'OU') | (crash.LICENSEPLATESTATE == 'VI') | (crash.LICENSEPLATESTATE == 'PU') | (crash.LICENSEPLATESTATE == 'UN')
crash = crash.where(plate_filter)

# Arianna end (all code written by me)
# Filter YEAR  < 2008 and YEAR > 2021
print("Before dropping data for YEAR < 2008 and YEAR > 2021")
print(crash.shape)
crash= crash[(crash.YEAR <= 2021) | (crash.YEAR >= 2016)]
print("Data shape after filtering YEAR")
print(crash.shape)

