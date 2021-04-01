# Group 3 Preprocessing

# Arianna Code Start:
import data_cleanup
# import other libraries below:

crash = data_cleanup.data
print(crash.shape)

# Filter YEAR  < 2008 and YEAR > 2021
print("Before dropping data for YEAR < 2008 and YEAR > 2021")
print(crash.shape)
crash = crash[crash.YEAR >= 2008]
print("Data shape after filtering YEAR < 2008")
print(crash.shape)
crash = crash[crash.YEAR <= 2021]
print(crash.shape) #(591121, 37)

# Checking to see the number of crashes involving someone 100+ to determine if that's a good
# Age to cap at
print("Crashes involving someone who is 100 or older:")
print(crash[crash.AGE >= 100.0].count()) # Result is 269

print(crash['AGE'].describe())
# Dropping all ages below 0 and above 100
age_filter = (crash.AGE > 100.0) | (crash.AGE < 0.0)
crash = crash.where(~age_filter)

print(crash['AGE'].describe())

# Dropping state abbreviations that do not exist (Ot, ou, vi, pu, un)
crash = crash.drop(crash[(crash.LICENSEPLATESTATE == 'Ot') | (crash.LICENSEPLATESTATE == 'Ou') | (crash.LICENSEPLATESTATE == 'Vi') | (crash.LICENSEPLATESTATE == 'Pu') | (crash.LICENSEPLATESTATE == 'Un') | (crash.LICENSEPLATESTATE == 'Am') | (crash.LICENSEPLATESTATE == 'Di')].index)


###
# RR - Create Feature Matrix and Fill/Drop NANs
###
# Create Matrix with only the feature rows we want and the target
print(crash.columns) # view columns
# drop redundent/unnecessary columns
crash_fm = crash.drop(['CRIMEID', 'REPORTDATE','MAJORINJURIES_BICYCLIST','MINORINJURIES_BICYCLIST','FATAL_BICYCLIST','MAJORINJURIES_DRIVER',
            'MINORINJURIES_DRIVER','FATAL_DRIVER','MAJORINJURIES_PEDESTRIAN','MINORINJURIES_PEDESTRIAN',
            'FATAL_PEDESTRIAN','MAJORINJURIESPASSENGER','FATALPASSENGER','MAJORINJURIESPASSENGER','IMPAIRED',
            'ROUTEID','DAY','HOUR','FATALMAJORINJURIES_TOTAL'], axis=1)
print(crash_fm.columns) # check the columns again

# Fill in missing data
print(crash_fm.isnull().sum()) # get missing values
crash_fm_age = crash_fm.drop(['AGE'], axis=1) # use to view rows with missing data, except age
null_data = crash_fm_age[crash_fm_age.isnull().any(axis=1)] # view rows with missing data
#print(null_data.head(50)) # view the null data rows
#print(crash_fm.head(20)) # view the head of the feature matrix
# it appears that there are simply 321 rows of empty data
# delete rows with completely empty data
crash_fm.dropna(subset = ["LATITUDE"], inplace=True) # drop all that don't have a lat/long - empty rows/correct values necessary for Lydia
print(crash_fm.isnull().sum()) # check to see if there are any more NANs, beside AGE - Result: 3 NAN in Ward, 159,253 NAN in Age



