import Preprocessing
import pandas as pd
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN
from math import pi


def dbscan(coords):
    # Washington DC area is 68.34 square miles
    # Reference: Google
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
    dfx = dfx[(dfx.FATALMAJORINJURIES == 1)]
    coordinates = dfx[['LATITUDE', 'LONGITUDE']].to_numpy()
    #write method for finding optimal epsilon and min_pts
    dc_clusters = dbscan(coordinates)
    plot_clusters(dc_clusters)



main()