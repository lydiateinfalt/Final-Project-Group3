import readdata
import pandas as pd
import numpy as np

data = readdata.crash_all

crash = pd.DataFrame(data, columns=['CRIMEID','REPORTDATE','LATITUDE', 'LONGITUDE', 'WARD', 'MAJORINJURIES_BICYCLIST',
                                    'MINORINJURIES_BICYCLIST', 'FATAL_BICYCLIST','AGE',
                                    'MAJORINJURIES_DRIVER', 'MINORINJURIES_DRIVER','FATAL_DRIVER',
                                    'MAJORINJURIES_PEDESTRIAN', 'MINORINJURIES_PEDESTRIAN','FATAL_PEDESTRIAN',
                                    'TOTAL_VEHICLES','TOTAL_BICYCLES','TOTAL_PEDESTRIANS','DRIVERSIMPAIRED',
                                    'PEDESTRIANSIMPAIRED','BICYCLISTSIMPAIRED','OFFINTERSECTION','MINORINJURIESPASSENGER',
                                    'FATALPASSENGER','MAJORINJURIESPASSENGER', 'INVEHICLETYPE','TICKETISSUED','LICENSEPLATESTATE',
                                    'IMPAIRED', 'SPEEDING'
                                    ])

# Add YEAR, MONTH, DAY columns based on REPORTDATE

crash['YEAR'] = crash['REPORTDATE'].str[0:4]
crash['MONTH'] = crash['REPORTDATE'].str[5:7]
crash['DAY'] = crash['REPORTDATE'].str[8:10]

# Add a new column FATALMAJORINJURIES if any of the following columns = 1

crash['FATALMAJORINJURIES'] = np.where(((crash.MAJORINJURIES_BICYCLIST == 1) | (crash.FATAL_BICYCLIST == 1) | (crash.MAJORINJURIES_DRIVER == 1) | (crash.FATAL_DRIVER == 1 ) | (crash.MAJORINJURIES_PEDESTRIAN == 1) | (crash.FATAL_PEDESTRIAN == 1 )), 1, 0)

print(crash.shape)
print(crash.dtypes)
crash.to_csv("crash.csv")