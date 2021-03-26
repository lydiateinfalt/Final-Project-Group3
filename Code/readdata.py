import pandas as pd
import numpy as np

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
crash_dc = pd.read_csv("/Users/ariannadunham/Desktop/Data Mining DATS 6103/Project/Crashes_in_DC.csv", parse_dates=['REPORTDATE'])
columns1 = crash_dc.dtypes

# Detailed crash report join with above table using CRIMEID
# Reference: https://opendata.dc.gov/datasets/crash-details-table
crash_dc_det = pd.read_csv("/Users/ariannadunham/Desktop/Data Mining DATS 6103/Project/Crash_Details_Table.csv")
columns2 = crash_dc_det.dtypes

# Analysis of each data set
print("Total number of rows in data set (crashes_in_dc): ", crash_dc.shape[0])

print("Total number of rows in data set (crashes_in_dc): ", crash_dc_det.shape[0])

# Drop any duplicate CRIMEID
crash_dc.drop_duplicates(subset=['CRIMEID'])

# Join two data sets and create a new df called crashAll "inner" is by default on
crash_all = pd.merge(crash_dc, crash_dc_det, on="CRIMEID")

# Display column names and data type
columns = crash_all.dtypes
print(crash_all.dtypes)
print("Total number of rows in data set: ", crash_all.shape[0])
print(crash_all.head(10))
print(crash_all.tail(10))


# =================================================================
# Analysis of impaired columns

impaired = pd.DataFrame(crash_all,columns=['IMPAIRED', 'PEDESTRIANSIMPAIRED', 'BICYCLISTSIMPAIRED','DRIVERSIMPAIRED'])
impaired['IMPAIRED_NUMBER'] = impaired['PEDESTRIANSIMPAIRED'] + impaired['BICYCLISTSIMPAIRED'] + impaired['DRIVERSIMPAIRED']
print(impaired[['PEDESTRIANSIMPAIRED','BICYCLISTSIMPAIRED','DRIVERSIMPAIRED']].aggregate(np.sum))
impaired['IMPAIRED'] = impaired['IMPAIRED'].map({'Y': 1, 'N':0})
df = pd.DataFrame(impaired[impaired['IMPAIRED'] != impaired['IMPAIRED_NUMBER']])
print("Number of rows impaired data discrepancy", df.shape[0])
#df.to_csv("impaired.csv")

# =================================================================
# Analysis of  speeding columns

speeding = pd.DataFrame(crash_all, columns=['SPEEDING', 'SPEEDING_INVOLVED'])
speeding['SPEEDING'] = speeding['SPEEDING'].map({'Y': 1, 'N':0})
df2 = pd.DataFrame(speeding[speeding['SPEEDING'] != speeding['SPEEDING_INVOLVED']])
print("Number of rows with speeding data discrepancy", df2.shape[0])
#df2.to_csv("speeding.csv")

# =================================================================
# Analysis of Unknown Injuries Columns

data = crash_all
print("#Total rows", data.shape[0])
data = data[data.UNKNOWNINJURIES_DRIVER == 0]
print("# Rows after removing unknown drivers", data.shape[0])
data = data[data.UNKNOWNINJURIES_BICYCLIST == 0]
print("# Rows after removing unknown bicyclist", data.shape[0])
data = data[data.UNKNOWNINJURIES_PEDESTRIAN == 0]
print("# Rows after removing unknown pedestrian", data.shape[0])
data = data[data.UNKNOWNINJURIESPASSENGER == 0]
print("# Rows after removing unknown passenger", data.shape[0])
print("Total number of columns", data.shape[1])
data = data.drop(columns = ['UNKNOWNINJURIES_DRIVER', 'UNKNOWNINJURIES_BICYCLIST','UNKNOWNINJURIES_PEDESTRIAN','UNKNOWNINJURIESPASSENGER'], axis = 1)
print("After dropping 4 columns, total: ", data.shape[1])

