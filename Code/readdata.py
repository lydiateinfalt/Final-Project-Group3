import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
# from azureml.core import Workspace, Dataset

# subscription_id = '8c28b136-66c9-4ae3-8eae-292d55569d3d'
# resource_group = 'DATS6103'
# workspace_name = 'DATS6103Lydia'

# workspace = Workspace(subscription_id, resource_group, workspace_name)

# crashDC = Dataset.get_by_name(workspace, name='DC Crashes')
# crashDC.to_pandas_dataframe()

# All crashes reported in DC
# Reference: https://opendata.dc.gov/datasets/crashes-in-dc?geometry=13.108%2C-75.996%2C-10.798%2C89.827
crash_dc = pd.read_csv("Crashes_in_DC.csv")
columns1 = crash_dc.dtypes

# Detailed crash report join with above table using CRIMEID
# Reference: https://opendata.dc.gov/datasets/crash-details-table
crash_dc_det = pd.read_csv("Crash_Details_Table.csv")
columns2 = crash_dc_det.dtypes

# Analysis of each data set
print("Total number of rows in data set (crashes_in_dc): ", crash_dc.shape[0])

print("Total number of rows in data set (crashes_in_dc): ", crash_dc_det.shape[0])
# Join two data sets and create a new df called crashAll "inner" is by default on
crash_all = pd.merge(crash_dc, crash_dc_det, on="CRIMEID")

# Display column names and data type
columns = crash_all.dtypes
print(crash_all.dtypes)
print("Total number of rows in data set: ", crash_all.shape[0])
print(crash_all.head(10))
print(crash_all.tail(10))

# Add YEAR, MONTH, DAY columns based on REPORTDATE

crash_all['YEAR'] = crash_all['REPORTDATE'].str[0:4]
crash_all['MONTH'] = crash_all['REPORTDATE'].str[5:7]
crash_all['DAY'] = crash_all['REPORTDATE'].str[8:10]
print()

fatalcrashes = crash_all[( crash_all['FATAL_BICYCLIST'] > 0) | ( crash_all['FATAL_DRIVER'] > 0) | ( crash_all['FATAL_PEDESTRIAN'] > 0)]
majorcrashes = crash_all[(crash_all['MAJORINJURIES_BICYCLIST'] > 0) | (crash_all['MAJORINJURIES_DRIVER'] > 0) | (crash_all['MAJORINJURIES_PEDESTRIAN'] > 0)]

f = pd.DataFrame(fatalcrashes.groupby(['YEAR']).size())
m = pd.DataFrame(majorcrashes.groupby(['YEAR']).size())
t= m + f

# plot number of accidents per year
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
ax.bar(m.index, m[0], color = "blue")
ax.bar(f.index, f[0], color = "red")

labels = ("Major Injuries", "Fatalities")
ax.set_xlabel('Year')
ax.set_title('Number of Crashes in DC Resulting in Fatalities/Major Injuries')
ax.legend(labels)
plt.show()


# No fatality, no major crashes
no_fatal_crashes = crash_all[(crash_all['FATAL_BICYCLIST'] == 0) | (crash_all['FATAL_DRIVER'] == 0) | (crash_all['FATAL_PEDESTRIAN'] == 0)]
no_major_crashes = crash_all[(crash_all['MAJORINJURIES_BICYCLIST'] == 0 ) | (crash_all['MAJORINJURIES_DRIVER'] == 0) | (crash_all['MAJORINJURIES_PEDESTRIAN'] == 0)]

f0 = pd.DataFrame(no_fatal_crashes.groupby(['YEAR']).size())
m0 = pd.DataFrame(no_major_crashes.groupby(['YEAR']).size())
t0 = m0 + f0
fig, ax = plt.subplots()
fig.set_size_inches(16,12)
ax.bar(t0.index, f0[0], color = "green")
ax.set_xlabel('Year')
ax.set_title('Number of Crashes Without Fatalities')
plt.show()
