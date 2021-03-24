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

# Use this to make sure you incorporate all years contained within the data - there are some odd years
# Get the total number of crashes per year, visually - RR 6 lines, 6 myself
year_counts = crash["YEAR"].value_counts() # get total counts of fatal/majorinjury and none/minorinjury
year_proportions = crash["YEAR"].value_counts(normalize=True)  # get the proportions
print('The number of crashes in DC per year are:')
print(year_counts)  # print total numbers
print('The proportion of crashes in DC by year are:')
print(year_proportions)  # print proportions

# Plot crashes by year - RR 10 copied, 7 copied, 5 modified, 3 my own
fig, ax = plt.subplots()
crash["YEAR"].plot.hist(ax=ax, bins = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025], edgecolor='black')
plt.xlim(xmin=2000, xmax = 2025)
labels = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]
#labels1 = np.arange(2000,2026)
ax.set_xticks(labels)#.tolist()
ax.set_xticklabels(labels, rotation=90)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Crashes")
ax.set_title('Crashes by Year')
plt.show()


# Use this to make sure you incorporate all months contained within the data  and to check for any oddities
# Get the total number of crashes per year, visually - RR 6 lines, 6 myself
year_counts = crash["MONTH"].value_counts() # get total counts of fatal/majorinjury and none/minorinjury
year_proportions = crash["MONTH"].value_counts(normalize=True)  # get the proportions
print('The number of crashes in DC per month are:')
print(year_counts)  # print total numbers
print('The proportion of crashes in DC by month are:')
print(year_proportions)  # print proportions


# Plot crashes by month - RR 10 copied, 7 copied, 5 modified, 3 my own
fig, ax = plt.subplots()
crash["MONTH"].astype(np.int64).plot.hist(ax=ax, bins = [1,2,3,4,5,6,7,8,9,10,11,12,13], color = 'orange', edgecolor='black')
plt.xlim(xmin= 1, xmax = 13)
labels_month = [1,2,3,4,5,6,7,8,9,10,11,12]
ax.set_xticks(labels_month)#.tolist()
ax.set_xticklabels(labels_month, rotation=90)
#labels = ax.get_xticks().tolist()
#labels = pd.to_datetime(labels)
#ax.set_xticklabels(labels, rotation=45)
ax.set_xlabel("Month")
ax.set_ylabel("Number of Crashes")
ax.set_title('Crashes by Month')
plt.show()


# Citation: "Matplotlib.axes.Axes.set_xticks() in Python". GeeksforGeeks. April 19, 2020. https://www.geeksforgeeks.org/matplotlib-axes-axes-set_xticks-in-python/

### ------------------------------------------------------------------------------------------
###
### This plot displays a map of DC with fatal/majorinjyr crashes overlaid
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

