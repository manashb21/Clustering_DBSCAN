import numpy as np
import random
from shapely.geometry import Polygon,Point
import pandas as pd 
#import h3
import folium

import matplotlib.pyplot as plt

poly = Polygon([(27.741969, 85.333064),
                (27.722305, 85.292919),
                (27.688432, 85.285059),
                (27.657974, 85.322721),
                (27.677999, 85.349656)])

#Defining the randomization generator
def polygon_random_points(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds
    
    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)
    return points

points = polygon_random_points(poly,1200)

latitude = []
longitude = []

# Printing the results.
for p in points:
    latitude.append(p.x)
    longitude.append(p.y)

df = pd.DataFrame()
df['Latitude'] = latitude
df['Longitude'] = longitude

plt.plot(df['Latitude'], df['Longitude'], marker = '.', linewidth = 0, color = 'turquoise')


map1 = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()], zoom_start=16, control_scale=True)

for index, location_info in df.iterrows():
    folium.CircleMarker([location_info["Latitude"], location_info["Longitude"]], radius = 4,color = 'black').add_to(map1)
    
map1    

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def dbscanalg(minpts, silhouette_score):
    kms_per_radian = 6371.0088
    epsilon = 0.2/ kms_per_radian
    clustering = DBSCAN(eps=epsilon, min_samples=minpts, algorithm='ball_tree', metric='haversine').fit(np.radians(df))
    cluster_labels = clustering.labels_
    num_clusters = len(set(cluster_labels))
    print(silhouette_score)
    print("The number of clusters: " ,num_clusters)
    return cluster_labels
    


minpts = 1
cluster_labels = dbscanalg(minpts, silhouette_score) 

while silhouette_score(df, cluster_labels) <0:
    minpts = minpts+1
    dbscanalg(minpts, silhouette_score(df, cluster_labels))
 

print(silhouette_score(df, cluster_labels))
    

outliers_df = df[cluster_labels == -1]
clusters_df = df[cluster_labels != -1]

colors = cluster_labels
color_clusters = colors[colors!=-1]
color_outliers = 'white'

len(np.unique(color_clusters))

fig = plt.figure(figsize = (5,5))

ax = fig.add_axes([.1,.1,1,1])

ax.scatter(clusters_df['Latitude'], clusters_df['Longitude'], c = color_clusters, edgecolor = 'black', s = 50)
ax.scatter(outliers_df['Latitude'], outliers_df['Longitude'], c = color_outliers, edgecolor = 'black', s = 50)

plt.show()
