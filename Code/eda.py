# DC Crash Data EDA
# RyeAnne Ricker

# import libraries
import data_cleanup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
#from shapely.geometry import Point, Polygon
#from scipy import stats
import researchpy as rp
import seaborn as sb
import seaborn as sns

# to call from my directory RR 2 lines, 2 myself
#crash = pd.read_csv('/Users/RyeAnne/Documents/School/Spring2021/DataMining/Group Project/crash_ver1.csv') # load data
# may use the following line to call it from class instead
crash = data_cleanup.data
print(crash.columns)

### ------------------------------------------------------------------------------------------
###
### This part displays histograms of crashes by year and month
###

# convert to datetime - RR 1 copied  - unnecessary due to Lydias conversion
#crash["REPORTDATE"] = crash["REPORTDATE"].astype("datetime64")

# This one is unneeded since Lydia converted the date to columns of year and month
# Plot crashes by year - RR 10 copied, 5 modified
#fig, ax = plt.subplots()
#crash["REPORTDATE"].dt.year.astype(np.int64).plot.hist(ax=ax, bins = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025], edgecolor='black')
#plt.xlim(xmin=2000, xmax = 2025)
#labels = ax.get_xticks().tolist()
#labels = pd.to_datetime(labels)
#ax.set_xticklabels(labels, rotation=90)
#ax.set_xlabel("Year")
#ax.set_ylabel("Number of Crashes")
#ax.set_title('Crashes by Year')
#plt.show()

# Use this to make sure you incorporate all years contained within the data  and to check for any oddities
# Get the total number of crashes per year, visually - RR 6 lines, 6 myself
year_counts = crash["MONTH"].value_counts() # get total counts of fatal/majorinjury and none/minorinjury
year_proportions = crash["MONTH"].value_counts(normalize=True)  # get the proportions
print('The number of crashes in DC per month are:')
print(year_counts)  # print total numbers
print('The proportion of crashes in DC by month are:')
print(year_proportions)  # print proportions

# Plot crashes by year - RR 4 copied, 4 modified, 3 my own
year_count = sns.countplot(data=crash, x='YEAR', hue='FATALMAJORINJURIES', dodge=False, palette="Paired")
year_count.set_xticklabels(year_count.get_xticklabels(), rotation=70)
year_count.set_xlabel("Year")
year_count.set_ylabel("Number of Crashes")
year_count.set_title('Crashes by Year')
year_count.legend(bbox_to_anchor=(0.1, 0.9), loc=2, borderaxespad=0.)
plt.show()

# Use this to make sure you incorporate all months contained within the data  and to check for any oddities
# Get the total number of crashes per year, visually - RR 6 lines, 6 myself
year_counts = crash["MONTH"].value_counts() # get total counts of fatal/majorinjury and none/minorinjury
year_proportions = crash["MONTH"].value_counts(normalize=True)  # get the proportions
print('The number of crashes in DC per month are:')
print(year_counts)  # print total numbers
print('The proportion of crashes in DC by month are:')
print(year_proportions)  # print proportions

# Plot crashes by month - RR 4 copied, 4 modified, 3 my own
month_count = sns.countplot(data=crash, x='MONTH', hue='FATALMAJORINJURIES', dodge=False,palette="Paired")
month_count.set_xticklabels(month_count.get_xticklabels(), rotation=70)
month_count.set_xlabel("Month")
month_count.set_ylabel("Number of Crashes")
month_count.set_title('Crashes by Month')
month_count.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
plt.show()

# Citation: "Visualizing Distributions of Data". Seaborn. [Accesed Online: Mar 25, 2021]. https://seaborn.pydata.org/tutorial/distributions.html


# Heatmap of crashes - LNT
# Reference: https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c
df = pd.DataFrame(crash, columns=['REPORTDATE','FATALMAJORINJURIES'])
df['REPORTDATE'] = pd.to_datetime(df['REPORTDATE'])
df.set_index(['REPORTDATE'], inplace=True)
df = df[df.FATALMAJORINJURIES == 1]
df['MONTH'] = [i.month for i in df.index]
df['YEAR'] = [i.year for i in df.index]
# group by month and year
df = df.groupby(['MONTH', 'YEAR']).count()
df = df.unstack(level=0)

fig, ax = plt.subplots(figsize=(11, 9))
sb.heatmap(df,cmap="Blues",linewidth=0.3)

ax.xaxis.tick_top()
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(np.arange(12) + .5, labels=labels)

plt.xlabel('')
plt.ylabel('')
plt.title('Fatal/Major Injuries Traffic Accidents in Washington DC from 2000-2021')
plt.show()

# Citation: "Matplotlib.axes.Axes.set_xticks() in Python". GeeksforGeeks. April 19, 2020. https://www.geeksforgeeks.org/matplotlib-axes-axes-set_xticks-in-python/

### ------------------------------------------------------------------------------------------
###
### This plot displays a map of DC with fatal/majorinjury crashes overlaid
###

# Make geomap to plot where crashes occurred - RR  12 lines, 5 copied, 7 myself
# change this to directory for boundary file in directory
# dc_shape = gpd.read_file('/Users/RyeAnne/Documents/School/Spring2021/DataMining/Group Project/Washington_DC_Boundary.shp')
dc_shape = gpd.read_file("Washington_DC_Boundary.shp")
# # Make Geoplot of Fatal and NonFatal car crashes in DC - RR 11 copied and modified, 5 wrote own
crs = {'init':'epsg:4326'}
geometry=gpd.points_from_xy(crash.LONGITUDE, crash.LATITUDE)
#geometry = [Point(xy) for xy in zip(crash["LONGITUDE"],crash['LATITUDE'])]
gdf = gpd.GeoDataFrame(crash, crs=crs, geometry=geometry)
# #print(gdf.head())
fig,ax = plt.subplots(figsize = (15,15))
dc_shape.plot(ax=ax, color = 'grey',alpha=0.5,zorder=1)
#plt.show()
gdf[gdf['FATALMAJORINJURIES'] == 1].plot(ax = ax, markersize = 10, color = 'red', marker = '*',label='Fatal/Major Injury',zorder=2)
# # plot specifications
plt.title('Crash Fatalities and Major Injuries by GeoLocation')
plt.legend(bbox_to_anchor=(1.0, .5), prop={'size': 8})
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# Citation: Steward, Ryan. "GeoPandas 101: Plot any data with a latitude ad longitude on a map". Oct 31, 2018. https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972

# Get total number of major injuries/fatalities and minor injuries/no injuries
# RR  6 lines, 6 myself
fatal_counts = crash["FATALMAJORINJURIES"].value_counts() # get total counts of fatal/majorinjury and none/minorinjury
fatal_proportions = crash["FATALMAJORINJURIES"].value_counts(normalize=True)  # get the proportions
print('The number of fatal/major injury and none/minor injury crashes in DC are:')
print(fatal_counts)  # print total numbers
print('The proportion of fatal/major injury and none/minor injury crashes in DC are:')
print(fatal_proportions)  # print proportions

### ------------------------------------------------------------------------------------------
###
### This part looks at the feature values/categories
###

# RR 18 lines, 18 myself
print('The columns of the file are:')
print(crash.columns)

# Feature - Ward
ward_counts = crash["WARD"].value_counts()
print(ward_counts)
# Feature - Age
age_counts = crash["AGE"].value_counts()
print(age_counts)
# Feature - Total_Vehicles
vehicle_counts = crash["TOTAL_VEHICLES"].value_counts()
print(vehicle_counts)
# Feature -  Total_Bicycles
bicycle_counts = crash["TOTAL_BICYCLES"].value_counts()
print(bicycle_counts)
# Feature - Total_Pedestrians
pedestrian_counts = crash["TOTAL_PEDESTRIANS"].value_counts()
print(pedestrian_counts)
# Feature - DriversImpaired
driversimpaired_counts = crash["DRIVERSIMPAIRED"].value_counts()
print(driversimpaired_counts)
# Feature - PedestriansImpaired
pedestriansimpaired_counts = crash["PEDESTRIANSIMPAIRED"].value_counts()
print(pedestriansimpaired_counts)
# Feature - BicyclistsImpaired
bicyclistsimpaired_counts = crash["BICYCLISTSIMPAIRED"].value_counts()
print(bicyclistsimpaired_counts)
# Feature - OffIntersection
offintersection_counts = crash["OFFINTERSECTION"].value_counts()
print(offintersection_counts)
# Feature - InVehicleType
vehicletype_counts = crash["INVEHICLETYPE"].value_counts()
print(vehicletype_counts)
# Feature - TicketIssued
ticketissued_counts = crash["TICKETISSUED"].value_counts()
print(ticketissued_counts)
# Feature - LicensePlateState
licenseplatestate_counts = crash["LICENSEPLATESTATE"].value_counts()
print(licenseplatestate_counts)
# Feature - Impaired
impaired_counts = crash["IMPAIRED"].value_counts()
print(impaired_counts)
# Feature - Speeding
speeding_counts = crash["SPEEDING"].value_counts()
print(speeding_counts)
# Feature - YEAR
year_counts = crash["YEAR"].value_counts()
print(year_counts)
# Feature - MONTH
month_counts = crash["MONTH"].value_counts()
print(month_counts)

# In the Data PreProcessing, we need to decide what to do with the weird ages
# get rid of the 2025 data (2 data points)
# and figure out what to do with the Pu and Am license plates (2 data points)

# This portion determines whether features are independent of crash having a fatality/major injury
# Chi squared tests are performed on each categorical variables to determine independence

# RR - 30 lines, 15 copied and modified, 15 myself
# Ward
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["WARD"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Ward are:',test_results)
# Age
#crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["AGE"],test= "chi-square",expected_freqs= True,prop= "cell")
#print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Age are:',test_results)
# Total Vehicles
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TOTAL_VEHICLES"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Total Vehicles are:',test_results)
# Total Bicycles
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TOTAL_BICYCLES"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Total Bicycles are:',test_results)
# Total Pedestrians
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TOTAL_PEDESTRIANS"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Total Pedestrians are:',test_results)
# Drivers Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["DRIVERSIMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired Drivers are:',test_results)
# Pedestrians Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["PEDESTRIANSIMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired Pedestrians are:',test_results)
# Bicyclists Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["BICYCLISTSIMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired Bicyclists are:',test_results)
# Intersection
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["OFFINTERSECTION"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Intersection are:',test_results)
# Vehicle Type
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["INVEHICLETYPE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Vehicle Type are:',test_results)
# Ticket Issued
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TICKETISSUED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Tickets Issued are:',test_results)
# Lice Plate State
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["LICENSEPLATESTATE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and License Plate State are:',test_results)
# Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["IMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired are:',test_results)
# Speeding
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["SPEEDING"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Speeding are:',test_results)
# Year
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["YEAR"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Year are:',test_results)
# Month
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["MONTH"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Month are:',test_results)

# Arianna Start.

# Getting major injuries and fatalities by ward. 3/3 lines written by me
ward_mf = crash.groupby('WARD').agg({'FATALMAJORINJURIES_TOTAL':'sum'})
print ("Major injuries and fatalities by ward: ")
print(ward_mf)

# Getting a bar graph of the results. 5/5 lines written by me.
ward_mf_bar = ward_mf.plot.bar(figsize=(20, 10))
plt.ylabel('Major Injuries and Fatalities')
plt.xlabel('Ward')
plt.title('Major Injuries and Fatalities by Ward')
plt.show()
# A note about the results: ward two has the most major injuries and fatalities


# Getting summary stats for age. 3/3 lines written by me
age_stats = crash['AGE'].describe()
print("The summary statistics for age are: ")
print(age_stats)

# Based on the results of this, this column needs some cleaning (before cleaning max was 237 and min was -7990)
# Deleting rows where age > 122 and where age < 0
# I'm picking 122 as the max because that's the oldest age on record. 2/2 lines written by me
age_filter = (crash.AGE > 122.0) | (crash.AGE < 0.0)
crash = crash.where(~age_filter)


# Printing sum stats again to compare. 2/2 line written by me
print("The summary statistics for age(cleaned) are:")
print(crash['AGE'].describe())

# Getting average age and whether or not the accident resulted in major injury or fatality
# to see if there's any discrepancy from total average. 3/3 lines written by me
mf_age = crash.groupby('FATALMAJORINJURIES').agg({'AGE': 'mean'})
print("Average age involved in accidents with fatalities and major injuries: ")
print(mf_age)

# Note about results: about the same

# Getting a histogram of age 5/5 lines written by me
age_hist = plt.hist(crash['AGE'])
plt.ylabel('Count')
plt.xlabel('Age')
plt.title('Age of People Involved in Traffic Accidents')
plt.show()

# Getting a histogram of age and accidents with fatality/major injury. 7/7 lines written by me
mf_filter = crash[crash.FATALMAJORINJURIES.eq(1.0)]
age_mf_hist = plt.hist(mf_filter['AGE'])
plt.ylabel('Count')
plt.xlabel('Age')
plt.title('Age of People Involved in Traffic Accidents with Fatalities or Major Injuries')
plt.show()

# Getting the count of crashes with major/fatal per state. 2/2 by me
states = crash.groupby('LICENSEPLATESTATE').agg({'FATALMAJORINJURIES':'sum'})
print("Crashes per License Plate State:")
print(states)

# Counting total number of crashes from someone with a plate in the DMV and not in the DMV. 13/13 by me
dmv_crash = 0
non_dmv_crash = 0
no_plate = 0
for i in crash['LICENSEPLATESTATE']:
    if i == 'DC':
        dmv_crash += 1
    elif i == 'VA':
        dmv_crash += 1
    elif i == 'MD':
        dmv_crash += 1
    elif i == 'None':
        no_plate =+ 1
    else:
        non_dmv_crash += 1
print("Number of crashes from DMV plate: ")
print(dmv_crash)
print("Number of crahses from non-DMV Plate: ")
print(non_dmv_crash)

# Counting total number of crashes from someone with a plate in the DMV and not in the DMV resulting in major/fatal.
# the first 15 lines were remixed and the remainder were all written by me
# Link for remixed code:
# https://stackoverflow.com/questions/53153703/groupby-count-only-when-a-certain-value-is-present-in-one-of-the-column-in-panda
dc_mf = ((crash['LICENSEPLATESTATE'] == 'DC')
            .groupby([crash['FATALMAJORINJURIES']])
            .sum()
            .astype(int)
            .reset_index(name='count'))


va_mf = ((crash['LICENSEPLATESTATE'] == 'VA')
            .groupby([crash['FATALMAJORINJURIES']])
            .sum()
            .astype(int)
            .reset_index(name='count'))

md_mf = ((crash['LICENSEPLATESTATE'] == 'MD')
            .groupby([crash['FATALMAJORINJURIES']])
            .sum()
            .astype(int)
            .reset_index(name='count'))

dmv_mf = dc_mf.loc[1] + va_mf.loc[1] + md_mf.loc[1]
print("The total number of accidents with DMV plates resulting in fatalities or major injuries is: ")
print(dmv_mf.loc['count'])

fatalmajor = 0
for row in crash['FATALMAJORINJURIES']:
    if row == 1:
        fatalmajor += 1
    else:
        continue
non_dmv_mf = fatalmajor - dmv_mf.loc['count'] - no_plate
print("The total number of accident with plates outside of the DMV resulting in fatalities or major injuries is: ")
print(non_dmv_mf)
