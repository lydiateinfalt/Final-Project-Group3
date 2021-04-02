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
print('The list of columns before dropping any include')
print(crash.columns) # view columns
# drop redundent/unnecessary columns
crash_fm = crash.drop(['CRIMEID', 'REPORTDATE','MAJORINJURIES_BICYCLIST','MINORINJURIES_BICYCLIST','FATAL_BICYCLIST','MAJORINJURIES_DRIVER',
            'MINORINJURIES_DRIVER','FATAL_DRIVER','MAJORINJURIES_PEDESTRIAN','MINORINJURIES_PEDESTRIAN',
            'FATAL_PEDESTRIAN','MAJORINJURIESPASSENGER','FATALPASSENGER','MAJORINJURIESPASSENGER','IMPAIRED',
            'ROUTEID','DAY','HOUR','MINORINJURIESPASSENGER','FATALMAJORINJURIES_TOTAL'], axis=1)
print('The list of columns after keeping only those that we want to use as features include')
print(crash_fm.columns) # check the columns again

# Check data types of variables to make sure they are correct
print('The data types of each of the features is') # change offintersection to categorical
print(crash_fm.dtypes)
crash_fm["OFFINTERSECTION"].astype('category') # change intersection to a categorical variable


# Fill in missing data/drop
print('The number of empty values per column is')
print(crash_fm.isnull().sum()) # get missing values
crash_fm_age = crash_fm.drop(['AGE'], axis=1) # use to view rows with missing data, except age
null_data = crash_fm_age[crash_fm_age.isnull().any(axis=1)] # view rows with missing data
print(null_data.head(50)) # view the null data rows - Weirdly, they are completely empty - get rid of all the ones missing Lat and Ward
### NOTE: USE IF DR JAFARI THINKS WE SHOULD DROP AGE AS WELL AS THE EMPTY ROWS
crash_fm.dropna(subset = ["LATITUDE"], inplace=True) # drop Lat and Ward but keep age
# crash_fm.dropna(subset = ["AGE"], inplace = True) # Use if Dr Jafari says 28% of missing values is too much and tells us to delet ethem
crash_fm.AGE.fillna(crash_fm.AGE.mean(), inplace=True)  # impute empty age cells with the average value
ward_replace = crash_fm['WARD'].value_counts().idxmax() # impute empty cells for WARD using the most common category
crash_fm.WARD.fillna(ward_replace, inplace=True) # impute WARD cells with most common ward
print('The number of empty values per column after imputation is:')
print(crash_fm.isnull().sum())  # check to make sure there is no longer empty cells




# write to csv file
crash_fm.to_csv("fm.csv")



