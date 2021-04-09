# Group 3 Preprocessing

# Import libraries
import readdata
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
# import other libraries below:

# Arianna Code Start:
crash = readdata.crash
print(crash.shape)

### NOTE: Due to the change in data sets (only using Crash Details now) we no longer have date information
# # Filter YEAR  < 2008 and YEAR > 2021
# print("Before dropping data for YEAR < 2008 and YEAR > 2021")
# print(crash.shape)
# crash = crash[crash.YEAR >= 2008]
# print("Data shape after filtering YEAR < 2008")
# print(crash.shape)
# crash = crash[crash.YEAR <= 2021]
# print(crash.shape) #(591121, 37)

# Checking to see the number of crashes involving someone 100+ to determine if that's a good
# Age to cap at
print("Crashes involving someone who is 100 or older:")
print(crash[crash.AGE >= 100.0].count()) # Result is 271

print(crash['AGE'].describe())
# Dropping all ages below 0 and above 100
age_filter = (crash.AGE > 100.0) | (crash.AGE < 0.0)
crash = crash.where(~age_filter)

print(crash['AGE'].describe())

# Dropping state abbreviations that do not exist (Ot, ou, vi, pu, un)
crash = crash.drop(crash[(crash.LICENSEPLATESTATE == 'Ot') | (crash.LICENSEPLATESTATE == 'Ou') | (crash.LICENSEPLATESTATE == 'Vi') | (crash.LICENSEPLATESTATE == 'Pu') | (crash.LICENSEPLATESTATE == 'Un') | (crash.LICENSEPLATESTATE == 'Am') | (crash.LICENSEPLATESTATE == 'Di')].index)


###
# RR - Create Feature Matrix and Fill/Drop NANs - 29 lines, 17 I wrote, 12 copied, 12 modified
###
# Create Matrix with only the feature rows we want and the target
print('The list of columns before dropping any include:')
print(crash.columns) # view columns
# drop redundent/unnecessary columns
crash_fm = crash.drop(['PERSONID','FATAL','MAJORINJURY','MINORINJURY'], axis=1)
print('The list of columns after keeping only those that we want to use as features include:')
print(crash_fm.columns) # check the columns again

# Fill in missing data
print('The number of empty values per column is')
print(crash_fm.isnull().sum()) # get missing values
crash_fm_age = crash_fm.drop(['AGE'], axis=1) # use to view rows with missing data, except age
null_data = crash_fm_age[crash_fm_age.isnull().any(axis=1)] # view rows with missing data
print(null_data.head(50)) # view the null data rows - Weirdly, they are completely empty - delete 328 empty rows
crash_fm.dropna(subset = ["PERSONTYPE"], inplace=True) # rows missing PERSONTYPE-these will delete all of the empty 328 rows
# crash_fm.dropna(subset = ["AGE"], inplace = True) # Use if Dr Jafari says 28% of missing values is too much and tells us to delet ethem
crash_fm.AGE.fillna(crash_fm.AGE.mean(), inplace=True)  # impute empty age cells with the average value
print('The number of empty values per column after imputation is:')
print(crash_fm.isnull().sum())  # check to make sure there is no longer empty cells - all dealt with
# deal with nan values-there are none
print(crash_fm.groupby(by='IMPAIRED').agg('count'))
print(crash_fm.groupby(by='TICKETISSUED').agg('count'))
print(crash_fm.groupby(by='INVEHICLETYPE').agg('count'))
print(crash_fm.groupby(by='SPEEDING').agg('count'))
print(crash_fm.groupby(by='FATALMAJORINJURIES').agg('count'))



# Label Encoder
ord_enc = OrdinalEncoder()
crash_fm["PERSONTYPE"] = ord_enc.fit_transform(crash_fm[["PERSONTYPE"]])
crash_fm["TICKETISSUED"] = ord_enc.fit_transform(crash_fm[["TICKETISSUED"]])
crash_fm["INVEHICLETYPE"] = ord_enc.fit_transform(crash_fm[["INVEHICLETYPE"]])
crash_fm["SPEEDING"] = ord_enc.fit_transform(crash_fm[["SPEEDING"]])
crash_fm["LICENSEPLATESTATE"] = ord_enc.fit_transform(crash_fm[["LICENSEPLATESTATE"]])
crash_fm["IMPAIRED"] = ord_enc.fit_transform(crash_fm[["IMPAIRED"]])
print('The new label encoded feature matrix is:')
print(crash_fm.head(11))

print('The data types of each feature after reassignment is:')
print(crash_fm.dtypes)

# Data Normalization - for use on Age
cols_to_norm = ['AGE']
crash_fm[cols_to_norm] = StandardScaler().fit_transform(crash_fm[cols_to_norm])
# manual normalization
# normalize age
# mean_age = crash_fm.AGE.mean()
# max_age = crash_fm.AGE.max()
# min_age = crash_fm.AGE.min()
# crash_fm['AGE'] = crash_fm['AGE'].apply(lambda x: (x - mean_age ) / (max_age -min_age)) # normalize
# # Citation: StackOverflow. https://stackoverflow.com/questions/28576540/how-can-i-normalize-the-data-in-a-range-of-columns-in-my-pandas-dataframe/28577480
print('Final Check of the final data:')
print(crash_fm.head(20))

# final feature matrix to call
crash_model = crash_fm
crash_model.to_csv("crash_model.csv")
### RR End
### The output is the cleaned, filled, and normalized matrix containing only the features and the target

