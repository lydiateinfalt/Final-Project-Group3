import readdata
import pandas as pd
import numpy as np

data = readdata.crash_all

data = pd.DataFrame(data, columns=['CRIMEID','REPORTDATE','LATITUDE', 'LONGITUDE', 'WARD', 'MAJORINJURIES_BICYCLIST',
                                    'MINORINJURIES_BICYCLIST', 'FATAL_BICYCLIST','AGE',
                                    'MAJORINJURIES_DRIVER', 'MINORINJURIES_DRIVER','FATAL_DRIVER',
                                    'MAJORINJURIES_PEDESTRIAN', 'MINORINJURIES_PEDESTRIAN','FATAL_PEDESTRIAN',
                                    'TOTAL_VEHICLES','TOTAL_BICYCLES','TOTAL_PEDESTRIANS','DRIVERSIMPAIRED',
                                    'PEDESTRIANSIMPAIRED','BICYCLISTSIMPAIRED','OFFINTERSECTION','MINORINJURIESPASSENGER',
                                    'FATALPASSENGER','MAJORINJURIESPASSENGER', 'INVEHICLETYPE','TICKETISSUED','LICENSEPLATESTATE',
                                    'IMPAIRED', 'SPEEDING', 'ROUTEID'
                                    ])

# Add YEAR, MONTH, DAY columns based on REPORTDATE
print("Convert REPORTDATE into YEAR, MONTH, DAY")
data['REPORTDATE'] = pd.to_datetime(data['REPORTDATE'])
data['YEAR'], data['MONTH'], data['DAY'], data['HOUR'] = data['REPORTDATE'].dt.year, data['REPORTDATE'].dt.month, data['REPORTDATE'].dt.day, data['REPORTDATE'].dt.hour

# Remove data older than 2000
data = data[data.YEAR > 1999]

# Add a new column FATALMAJORINJURIES if any of the following columns = 1
data['FATALMAJORINJURIES'] = np.where(((data.MAJORINJURIES_BICYCLIST > 0) | (data.FATAL_BICYCLIST > 0) | (data.MAJORINJURIES_DRIVER > 0) | (data.FATAL_DRIVER > 0) | (data.MAJORINJURIES_PEDESTRIAN > 0) | (data.FATAL_PEDESTRIAN > 0) | (data.MAJORINJURIESPASSENGER > 0) | (data.FATALPASSENGER > 0)), 1, 0)
data['FATALMAJORINJURIES_TOTAL'] =  data['MAJORINJURIES_BICYCLIST'] + data['FATAL_BICYCLIST'] + data['MAJORINJURIES_DRIVER'] + data['FATAL_DRIVER'] + data['MAJORINJURIES_PEDESTRIAN'] + data['FATAL_PEDESTRIAN'] + data['MAJORINJURIESPASSENGER'] + data['FATALPASSENGER']

# After EDA, we can drop any unnecessary columns, for now keep Major/Minor/Fatal columns

print(data.shape)
print(data.dtypes)
data.to_csv("crash_ver1.csv")