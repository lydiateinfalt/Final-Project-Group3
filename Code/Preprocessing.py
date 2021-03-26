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
plate_filter = (crash.LICENSEPLATESTATE != 'OT') | (crash.LICENSEPLATESTATE != 'OU') | (crash.LICENSEPLATESTATE != 'VI') | (crash.LICENSEPLATESTATE != 'PU') | (crash.LICENSEPLATESTATE != 'UN')
crash = crash.where(plate_filter)



