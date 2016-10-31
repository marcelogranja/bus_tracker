# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:58:30 2016

@author: marcelo
"""

import utm
import math
import os
from random import randint
import sys
import pandas as pd
import pickle
import sqlite3
from sklearn.cluster import KMeans
from shapely.geometry import Point
from shapely.geometry import LineString
import matplotlib.pyplot as plt
#import ggplot


os.chdir('/home/marcelo/Desktop/2016.1/Data Incubator/challenge/q3/deploy')

busline = '583'

def calc_table():
    

    df = pd.read_csv('data' + busline + '.csv', header = 0, \
        names = ['timestamp', 'bus_id', 'line_id', 'latitude', 'longitude', 'speed'])
    
    locations = zip(df['latitude'].tolist(), df['longitude'].tolist())
    
    print 'size:', len(locations)

    # converts to UTM    
    E, N = [], []
    for idx, (lat, lon) in enumerate(locations):
        e, n, z, l = utm.from_latlon( lat, lon )
        E.append(e)
        N.append(n)
        if idx % 10000 == 0 :
            print idx            
    df['x'], df['y']= E, N

    # matches gps observations within 100m squared tiles
    
    x_min = df['x'].min()
    y_min = df['y'].min()
    
    df['tile_x'] = df['x'].apply( lambda x : int( math.floor( ( x - x_min) / TILE_SIZE ) ) )
    df['tile_y'] = df['y'].apply( lambda y : int( math.floor( ( y - y_min) / TILE_SIZE ) ) )
    
    df['tile'] = df.apply( (lambda x : (x['tile_x'], x['tile_y']) ), axis = 1)        
    df['tile'] = df['tile'].astype(str)
    
    df.to_sql('gps', con, if_exists = 'replace')
    pickle.dump( (TILE_SIZE, x_min, y_min), open('data' + busline + '.pck', 'wb') )
    
    print 'saved to db'
    sys.exit()

con = sqlite3.connect('test.db')


TILE_SIZE = float(100)
CACHE = True

if CACHE == False:
    calc_table()
else:
    sql = "\
    SELECT timestamp, bus_id, x, y, tile, speed \
    FROM gps \
    WHERE speed > 5 \
    ORDER BY bus_id, timestamp"    
    df = pd.read_sql_query(sql, con)
    (TILE_SIZE, x_min, y_min) = pickle.load( open('data' + busline + '.pck', 'rb') )


# gets 100 candidate route waypoints using k-means

data = df[['x','y']].values[:10000]   # initial test case <<<<<<<<<<<<<<<<<<<<<<<
kmeans = KMeans(n_clusters=100, n_init=1).fit(data)
candidate_points_coords = kmeans.cluster_centers_
route = pd.DataFrame(candidate_points_coords, columns=['x','y'])

# plots 3000 gps observations

df = df.head(3000)
ax = df.plot(kind='scatter', x='x', y='y', color='blue')
route.plot(kind='scatter', x='x', y='y', color='red', ax=ax)

candidate_points = []
for x,y in candidate_points_coords:
    candidate_points.append( Point(x, y) ) #surprisingly slow

# build route by greedly searching for the nearest candidate waypoint up to a certain distance

threshold_distance = 1700

unvisited = list(candidate_points)
visited = [unvisited.pop( randint(0, len(unvisited) ) ) ]
            
while len(unvisited) > 0:
    current_point = visited[-1]
    closest_distance, closest_point_idx = 999999, -1
    
    for idx, candidate_point in enumerate(unvisited):
        if candidate_point.distance(current_point) < closest_distance:
            closest_distance = candidate_point.distance(current_point)
            closest_point_idx = idx

    if closest_distance < threshold_distance :
        visited.append( unvisited.pop(closest_point_idx) )
    else:
        unvisited.pop(closest_point_idx)

# plots candidate waypoints and the  proposed bus trajectory

coords = [point.coords.xy for point in visited]
coords = map( lambda x : (x[0][0], x[1][0]) , coords)
X, Y = zip(*coords)

route.plot(kind='scatter', x='x', y='y')

plt.plot(X, Y)
plt.title('Candidate waypoints (blue points) and proposed bus trajectory (blue line)')
plt.show()


#builds route linestring and computes the location of vehicles along it

route_line = LineString(coords)

def project_location(r):
    current_point = Point(r['x'], r['y'])
    return route_line.project( current_point )/1000 , route_line.distance( current_point )

(df['km'], df['deviation']) = zip( *df.apply(project_location, axis=1) )

bus_ids = df['bus_id'].value_counts().index

# plots a time vs location diagram for vehicles along the proposed route

colors = ['r','y','g','b']
for idx, bus in enumerate(bus_ids[:3]):
    tmp = df[ df['bus_id'] == bus ]
    tmp = tmp[ tmp['deviation'] < 1000 ].head(100)
    if idx == 0:
        ax = tmp.plot(x='timestamp', y='km', color = colors[idx], legend=None, \
                        title = 'Location vs time for 3 buses travelling along the proposed route.')
    else:
        tmp.plot(x='timestamp', y='km', color = colors[idx], legend=None, ax=ax)
ax.legend().set_visible(False)
plt.show()
