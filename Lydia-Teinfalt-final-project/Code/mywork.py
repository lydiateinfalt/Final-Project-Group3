import Preprocessing
import pandas as pd
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN
from math import pi

import pandas as pd
import numpy as np

# Crashes reported in DC
# Reference: https://opendata.dc.gov/datasets/crash-details-table
crash = pd.read_csv('https://opendata.arcgis.com/datasets/70248b73c20f46b0a5ee895fc91d6222_25.csv')
#crash = pd.read_csv("Crash_Details_Table.csv")
# Analysis of each data set
print("Original data set")
print("Total number of rows in data set: ", crash.shape[0])
print("Total number of columns in data set:", crash.shape[1])

crash = pd.DataFrame(crash, columns=['PERSONID', 'PERSONTYPE', 'AGE', 'FATAL', 'MAJORINJURY', 'MINORINJURY', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING'])
print("Selecting columns from data set")
print("Total number of rows in data set: ", crash.shape[0])
print("Total number of columns in data set:", crash.shape[1])

crash['FATALMAJORINJURIES'] = np.where((crash['FATAL'].eq('Y') | crash['MAJORINJURY'].eq('Y')),1,0)
print("Adding new column FATALMAJORINJURIES")
print("Total number of rows in data set: ", crash.shape[0])
print("Total number of columns in data set:", crash.shape[1])
fatal_crash=crash[crash.FATALMAJORINJURIES.eq(1.0)]
print("Total number observations with FATALMAJORINJURES", fatal_crash.shape[0])
print("Percentage of FATAL crashes", (fatal_crash.shape[0]/crash.shape[0]*100))


print("Dataset first few rows:\n ")
print(crash.head())

print ('-'*80 + '\n')

# printing the structure of the dataset
print("Dataset info:\n ")
print(crash.info())
print ('-'*80 + '\n')
# printing the summary statistics of the dataset
print(crash.describe(include='all'))
print ('-'*80 + '\n')
print("Fatal crashes")
print(crash[crash['FATAL'] == 'Y'])
crash.to_csv("crash.csv")

def dbscan(coords):
    # Washington DC area is 68.34 square miles
    # Reference: Google
    # Reference: https://towardsdatascience.com/mapping-the-uks-traffic-accident-hotspots-632b1129057b
    earth_perimeter = 40070000.0  # In miles
    eps_in_radians = 500 / earth_perimeter * (2 * pi) #half a mile
    mininum = 600
    db = DBSCAN(eps=eps_in_radians, min_samples=mininum, metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    print("cluster_labels:", cluster_labels)
    print("num_clusters", num_clusters)
    clusters = pd.DataFrame([coords[cluster_labels==n] for n in range(num_clusters)])
    return clusters

def plot_clusters(dc_clusters):
    fig, ax = plt.subplots(figsize=(15, 15))

    ax = plt.gca()
    dc_shape = gpd.read_file("Roads.shp")
    dc_shape.plot(ax=ax, color='grey', alpha=0.7, zorder=1)
    num_clusters=len(dc_clusters)

    for i in range(0, num_clusters - 1):
        lats, longs = zip(*dc_clusters.iloc[i,0])
        ax.scatter(x=longs, y=lats, alpha=0.2, zorder=1, cmap='BuGn')
    plt.title('Clusters Identified by DBASE')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def main():
    # read in feature matrix
    # Fiona package cannot read zipfiles on Windows
    zipfile = "zip:///Roads.zip//Roads.shp"

    #preprocessed feature matrix (fm)
    crash = pd.read_csv("fm.csv")
    dc_shape = gpd.read_file("Roads.shp")

    dfx = pd.DataFrame(crash)
    #dfx = dfx[(dfx.FATALMAJORINJURIES == 1)]
    coordinates = dfx[['LATITUDE', 'LONGITUDE']].to_numpy()
    #write method for finding optimal epsilon and min_pts
    dc_clusters = dbscan(coordinates)
    plot_clusters(dc_clusters)



main()