
# Import libraries
import readdata
import researchpy as rp
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Arianna Code Start:
crash = readdata.crash
print(crash.shape)

# RR - 30 lines, 15 copied and modified, 15 myself

crash.dropna(subset = ["PERSONTYPE"], inplace=True) # rows missing PERSONTYPE-these will delete all of the empty 328 rows
print('Columns with missing values include:')
print(crash.isnull().sum()) # Age can be missing values as it is not categorical and thus will not have a chi-squared test on it

# This portion determines whether features are independent of crash having a fatality/major injury
# Chi squared tests are performed on each categorical variables to determine independence

# Age - decided not to put into groups and to instead use a quant variable
#crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["AGE"],test= "chi-square",expected_freqs= True,prop= "cell")
#print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Age are:',test_results)
# Vehicle Type
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["INVEHICLETYPE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Vehicle Type are:',test_results)
# Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["IMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired are:',test_results)
# Ticket Issued
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TICKETISSUED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Tickets Issued are:',test_results)
# Lice Plate State
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["LICENSEPLATESTATE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and License Plate State are:',test_results)
# Speeding
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["SPEEDING"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Speeding are:',test_results)
# Person Type
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["PERSONTYPE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Person Type are:',test_results)

# Age is our only quantitative variable - run summary statistics on it
# First remove weird ages (<0 and >100)
age_filter = (crash.AGE > 100.0) | (crash.AGE < 0.0)
crash = crash.where(~age_filter)
young_drivers = crash[ (crash['AGE'] < 10) & (crash['PERSONTYPE'] == 'Driver')].index
crash.drop(young_drivers, inplace = True)
print('Summary Age statistics are:')
print(crash.AGE.describe())
#print('The mode is:',statistics.mode(crash.AGE))

# Look at distributions of the age of drives for those crashes that result in Fatalities/Major Injuries and Not
sns.set(style="darkgrid")
sns.displot(crash, x="AGE", hue="FATALMAJORINJURIES", kind="kde", multiple="stack",palette="Paired",legend=False)
#sns.displot(data=fmi, x="AGE", color="skyblue", label="Fatal/Major Injury", kde=True)
#sns.displot(data=minor, x="AGE", color="green", label="Minor Injury", kde=True)
plt.legend(labels=['Fatal/Major Injury','Minor Injury'])
plt.title('Distribution of Age By Injury Group')
plt.show()

# Normalize Age so that we can compare, despite the class imbalance
sns.set(style="darkgrid")
sns.displot(crash, x="AGE", hue="FATALMAJORINJURIES", kind="kde", multiple="stack",common_norm=False,palette="Paired",legend=False)
#sns.displot(data=fmi, x="AGE", color="skyblue", label="Fatal/Major Injury", kde=True)
#sns.displot(data=minor, x="AGE", color="green", label="Minor Injury", kde=True)
plt.legend(labels=['Fatal/Major Injury','Minor Injury'])
plt.title('Normalized Age Distributions')
plt.show()

# Run t-test between the ages of those acquiring major injuries/fatalities and minor injuries
fmi = crash.loc[crash['FATALMAJORINJURIES'] == 1.0] # group 1 is those that had a fatal major injury
fmi.dropna(subset = ["AGE"], inplace=True) # drop rows with nan for stats analysis
#print('The number of individuals acquiring a fatality/major injury is', fmi.shape[0])
minor = crash.loc[crash['FATALMAJORINJURIES'] == 0.0] # group 2 is those that did not
minor.dropna(subset = ["AGE"], inplace=True) # drop rows with nan for stats analysis
#print('The number of individuals not acquiring a fatality/major injury is', minor.shape[0])
#print('The proportion of individuals in a DC crash that ends up dead or with a major injury is', fmi.shape[0]/crash.shape[0])

# calculate p-value
t,p = stats.ttest_ind(fmi.AGE, minor.AGE)
print('The p-value of the t-test is:', p)
print('The mean age of those who have a fatality or major injury is', np.mean(fmi.AGE))
print('The standard deviation of fatal/major injury is', np.std(fmi.AGE))
print('The mean age of those who have a minor injury is', np.mean(minor.AGE))
print('The standard deviation of minor injury is', np.std(minor.AGE))

