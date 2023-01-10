import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation
import json
import matplotlib.cm as cm
import matplotlib.colors as colors
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
import math
from pylab import figure, text, scatter, show
import seaborn as sns
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

import geocoder # import geocoder

import folium # plotting library

#scraping webpages
from bs4 import BeautifulSoup

#Clustering 
from sklearn.cluster import KMeans

from sklearn.metrics import pairwise

#from sklearn.decomposition import PCA

import foursquare

from scipy.stats import norm

from mpl_toolkits.mplot3d import Axes3D 

import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path

my_dir= Path().absolute()

get_ipython().system('mklink my_dir+"/" helper_functions.py helper_functions.py')

import helper_functions as hf

from yellowbrick.cluster import KElbowVisualizer


CLIENT_ID = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx' # Foursquare ID
CLIENT_SECRET = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' # Foursquare Secret
ACCESS_TOKEN = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' # Foursquare tocken
VERSION = '20180604'

def add_coors(address):
    geolocator = Nominatim(user_agent="ny_explorer")
    location = geolocator.geocode(address)
    if not location:
        return 1
    latitude = location.latitude
    longitude = location.longitude
    
    return latitude, longitude


#Retreive Toronto Neighborhoods and Boroughs from wiki page
def load_Toronto_data():
    toronto_data = requests.get('https://en.wikipedia.org/wiki/List_of_city-designated_neighbourhoods_in_Toronto').text   
    soup = BeautifulSoup(toronto_data, 'html.parser')
    data = []

    table=soup.find(attrs={"class": "wikitable sortable"})

    table_body = table.find('tbody')
    rows = table_body.find_all('tr')

    for row in rows:
        cols = row.find_all('td')
        cols = [c.text.strip() for c in cols]
        if cols:
                data.append([c for c in cols if c])
    tz=pd.DataFrame(data)

    tz.columns = ['Latitude','Longitude','Borough','Neighborhood']
    tz=tz[['Borough','Neighborhood','Latitude','Longitude']]
    tz.loc[:,'Latitude']=0.0
    tz.loc[:,'Longitude']=0.0
    return tz


# Retreive Berlin Neighborhoods and Boroughs from wiki page
def load_Berlin_data():
    berlin_data = requests.get('https://de.wikipedia.org/wiki/Verwaltungsgliederung_Berlins').text   
    soup = BeautifulSoup(berlin_data, 'html.parser')

    data = []
    tables = soup.findAll("table") 
    table_body = tables[2].find('tbody')
    rows = table_body.find_all('tr')

    for row in rows:
        cols = row.find_all('td')
        cols = [c.text.strip() for c in cols]
        if cols:
            data.append([c for c in cols if c])

    br=pd.DataFrame(data)

    br=br.drop([br.columns.values[0],br.columns.values[5]], axis=1)
    br.columns = ['Neighborhood','Borough','Latitude','Longitude']
    br.loc[:,'Latitude']=0.0
    br.loc[:,'Longitude']=0.0

    return br


def adjusted_rating(df, estimator):
    
    df_res=df[:]
    
    l = df['Likes'].reset_index(drop=True).values.astype(float)
    r = df['Rating'].reset_index(drop=True).values.astype(float)
    
    # adjust venue rating based on the normalizer for likes in the area
    df_res['Rating_adj'] = l / estimator * r
    df_res['Rating_adj'] = df_res['Rating_adj'].astype(float)
    
    return df_res


def plot_2D (df, K, cols, ms, c=False):

    labels = df[cols[3]] #k_means.labels_
       
    colors = df['Cluster Labels']
    colors = colors.replace(np.arange(len(c)),c)
    
    fig = plt.figure(figsize=(15,4),constrained_layout=False)
    
    ax1 = fig.add_subplot(121)
    ax1.scatter(df[cols[0]],  df[cols[2]], c=colors, marker=ms) #labels.astype(np.float))
    ax1.set_xlabel(cols[0])
    ax1.set_ylabel(cols[2])

    ax2 = fig.add_subplot(122)
    ax2.scatter(df[cols[1]], df[cols[2]], c= colors,marker=ms)

    ax2.set_xlabel(cols[1])
    ax2.set_ylabel(cols[2])
    
    return

def plot_3D (df, K, cols, ms, c=False):

    labels = df[cols[3]] #k_means.labels_
    
    colors = df['Cluster Labels']
    colors = colors.replace(np.arange(len(c)), c)
            
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[3])

    ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], c=colors, marker=ms)
    
    return


def viz_distros(viz, cols, titles, bins, title, c):

    fig = plt.figure(figsize=(15,4), constrained_layout=False)

    fig.suptitle(title, fontsize=14)
    
    ax1 = fig.add_subplot(131)
    ax1.hist(viz[cols[0]], bins=bins[0], color=c)
    ax1.set_xlabel(titles[0])
    ax1.set_ylabel('Frequency')

    ax2 = fig.add_subplot(132)
    ax2.hist(viz[cols[1]], bins=bins[1], color=c)
    ax2.set_xlabel(titles[1])
    ax2.set_ylabel('Frequency')

    ax3 = fig.add_subplot(133)
    ax3.hist(viz[cols[2]], bins=bins[2], color=c)
    ax3.set_xlabel(titles[2])
    ax3.set_ylabel('Frequency')

    plt.show()


# function that plots data point on the map for a given zoom
# using latitude and longitude values of a given adress
def plot_markers(neighborhoods, map_n, c, s):

# add markers to map
    for lat, lng, borough, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
        label = '{}, {}'.format(neighborhood, borough)
        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=s,
            popup=label,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=0.7,
            parse_html=False).add_to(map_n)    
    return map_n


# function that plots data point on the map for a given zoom
# using latitude and longitude values of a given adress
def plot_venues(neighborhoods, map_n, c, s):

# add markers to map
    for lat, lng, cat, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Category'], neighborhoods['Neighborhood']):
        label = '{}, {}'.format(neighborhood, cat)
        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=s,
            popup=label,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=0.7,
            parse_html=False).add_to(map_n)    
    return map_n

def cluster_data(neighborhoods, k, cols):
    
   #clustering = neighborhoods[['Latitude', 'Longitude']]
    neighborhoods_cl = neighborhoods[:]

    # run k-means clustering

    #initialize with k-means++ in a smart way to speed up convergence
    kmeans = KMeans(init="k-means++", n_clusters=k, n_init=20).fit(neighborhoods_cl[cols])

    #lets add cluster labels to Toronto neighborhoods
    neighborhoods_cl.insert(0, 'Cluster Labels', kmeans.labels_)

    #lets calculate SSE = sum of sqares of disrances between data points in a cluster and cluster centroid
    #d = {'Cluster Labels':np.arange(k),'c_1': kmeans.cluster_centers_[:,0], 'c_2': kmeans.cluster_centers_[:,1]}
   # cl_centroids = pd.DataFrame(data=d)
    #neighborhoods_cl=neighborhoods_cl.merge(cl_centroids, on=['Cluster Labels'], how='left')

    #return clustered data
    return neighborhoods_cl


# function that plots clustered data point on the map for a given zoom
# using latitude and longitude values of a given adress
def plot_clusters(map_clusters, neighborhoods_cl, kclusters, s, rainbow=False):
    
    # set color scheme for the clusters
    x = np.arange(kclusters)
    ys = [i + x + (i*x)**2 for i in range(kclusters)]
    
    #generate ramdon filling
    if not rainbow:
        
        colors_array =cm.rainbow(np.random.rand(kclusters))
        rainbow = [colors.rgb2hex(i) for i in colors_array]
        
    #generate ramdon contor
    colors_array_f =cm.rainbow(np.random.rand(kclusters))
    rainbow_f = [colors.rgb2hex(i) for i in colors_array_f]
    
    # add markers to the map
    markers_colors = []
    for lat, lon, poi, cluster in zip(neighborhoods_cl['Latitude'], neighborhoods_cl['Longitude'], neighborhoods_cl['Neighborhood'], neighborhoods_cl['Cluster Labels']):
        label = folium.Popup(str(poi).format("UTF-8") + ' Cluster ' + str(cluster+1), parse_html=True)
        folium.CircleMarker(
            [lat, lon],
            radius=s,
            popup=label,
            color=rainbow[cluster-1],
            fill=True,
            fill_color=rainbow_f[cluster-1],
            fill_opacity=0.7).add_to(map_clusters)
    return [rainbow, map_clusters]


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

    
def expore_venue(venue_id):
    url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)
    result = requests.get(url).json()
    
    a = 0
    
    try:
        result['response']['venue']
        
        # get venue rating 
        try:
            r = result['response']['venue']['rating']
        except:
            r = 0
            
        # get venue pricing cathegory        
        try:
            p = result['response']['venue']['prices']
        except:
            p = 0
        
        # get number of likes
        try:
            l = result['response']['venue']['likes']['count']
        except:
            l = 0
        
    except:
        
        a = json.dumps(result) + "\n" + 'Foursquare quota was exceeded, we are using simulated values'+"\n"
        l = round(random.random()*300,0)+1
        r = np.random.rand(1,int(l))*9
        r = r.mean()+1
        p = round(random.random()*3,0)+1
    
    return [r, p, l, a]


#get venues given radius and coordinates
def get_venues(Neighborhood, latitude, longitude, radius, limit):
    
   # print(latitude, longitude)
    url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius,limit)
    results = requests.get(url).json()     
    
    final_df = pd.DataFrame() 
    
    if results['response']:
        
        venues = results['response']
        # assign relevant part of JSON to venues
        venues = results['response']['venues']

        # tranform venues into a dataframe
        dataframe = json_normalize(venues)
        # print(dataframe.head())

        # keep only columns that include venue name, and anything that is associated with location
        filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
        dataframe_filtered = dataframe.loc[:, filtered_columns]

        # filter the category for each row
        dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

        # clean column names by keeping only last term
        dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

        # pretify columns
        dataframe_filtered = dataframe_filtered.rename(columns={"categories": "Category", "lat": "Latitude", "lng": "Longitude", 'name':'Name','address':'Address','city':'Borough'})
        
        try:
            dataframe_filtered = dataframe_filtered.dropna(subset=['Address'])

        # keep only meaningful columns
            columns=['Name', 'Category', 'Address','Borough', 'Latitude', 'Longitude','id']

            final_df = dataframe_filtered[columns]
        except:
            return  final_df
    else:
        print('No venues returned for ',Neighborhood)
            
    return  final_df                                                                                                            


#let's add rating, price cathegoty and the number of tips for venues in Toronto
def get_all_venues(df, r, l):
    
    df_total = pd.DataFrame(columns=['Name', 'Category', 'Address','Borough', 'Neighborhood', 'Latitude', 'Longitude','id','Rating', 'Price_cat','Likes'])

    a = 0
    
    for n in np.arange(len(df['Neighborhood'])):
        
        Neighborhood = df.loc[n, 'Neighborhood']
        latitude = df.loc[n, 'Latitude']
        longitude = df.loc[n, 'Longitude']
        
        #get venues in the Neighborhood given radius and coordinates
        df_venues = get_venues(Neighborhood, latitude, longitude, r, l)

        if not df_venues.empty:
            
            df_venues = df_venues.set_index('id')
            df_venues['Rating'] = 0
            df_venues['Price_cat'] = 0
            df_venues['Borough'] = df.loc[n,'Borough']
            df_venues['Neighborhood'] = df.loc[n, 'Neighborhood']

            for v in df_venues.index:
                
                [r, p, l, a] = expore_venue(v)
                df_venues.at[v, 'Rating'] = r
                df_venues.at[v, 'Price_cat'] = p
                df_venues.at[v, 'Likes'] = l

            df_total=df_total.append(df_venues.reset_index());
            
        else:
            print("Dataset is empty, result:",  df_venues)
    
    df_total = df_total.dropna(subset=['Category'], axis=0);
    
    if a!=0:
        print(a)

    return df_total


def calc_likeness(df, col, W):
    res = W * df[col[0]] + (1 - W) * df[col[1]]
    return res


def bar_plot(my_venues_grouped, all_cats, series_col, label_col, title, c):
    
    y_pos = np.arange(len(all_cats))
    
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(15, 15)

    #rating is adjusted by the proportion in total numner of tips
    h = my_venues_grouped[series_col[0]]
    df =pd.DataFrame({label_col[0]: all_cats, series_col[0]: h})
    df=df.sort_values(by=series_col[0], ascending=False)

    ax[0].barh(y_pos,df[series_col[0]], align='center',color=c)
    ax[0].set_yticks(y_pos)
    ax[0].set_yticklabels(df[label_col[0]])
    ax[0].invert_yaxis()  # labels read top-to-bottom
    ax[0].set_title(title[0])
    
    h = my_venues_grouped[series_col[1]]
    df =pd.DataFrame({label_col[1]: all_cats, series_col[1]: h})
    df=df.sort_values(by=series_col[1], ascending=False)

    ax[1].barh(y_pos,df[series_col[1]], align='center',color=c)
    ax[1].set_yticks(y_pos)
    ax[1].set_yticklabels(df[label_col[1]])
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_title(title[1])
    
    h = my_venues_grouped[series_col[2]]
    df = pd.DataFrame({label_col[2]: all_cats, series_col[2]: h})
    df = df.sort_values(by=series_col[2], ascending=False)

    ax[2].barh(y_pos,df[series_col[2]], align='center', color=c)
    ax[2].set_yticks(y_pos)
    ax[2].set_yticklabels(df[label_col[2]])
    ax[2].invert_yaxis()  # labels read top-to-bottom
    ax[2].set_title(title[2])

    plt.tight_layout()
    plt.show()

    
def plot_freq(data_all_grouped, col, title):
    
    cols = data_all_grouped.columns
    df_sp = data_all_grouped[[cols[0], cols[1], col]].head(20)
    xlabels = df_sp[col].values
    df_sp.set_index(col)
    fig, ax = plt.subplots()
    color_list = ['#5cb85c', '#5bc0de']
    df_sp.plot(kind='bar', figsize=(20, 8), color=color_list, ax=ax, width=0.8, fontsize=14)
    for rect in ax.patches:
        height = rect.get_height()
        ypos = rect.get_y() + height+0.1
        ax.text(rect.get_x() + rect.get_width()/2., ypos,
                '%s' % int(height) +'', ha='center', va='bottom',fontsize=14)
    plt.legend(['Toronto', 'Berlin'], fontsize=14)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xticklabels(xlabels)
    plt.tick_params(top=False, left=False, right=False, labelleft=False)
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)  
    ax.set_title(title, fontdict={'fontsize': 16})
    plt.show()
    

def calc_freq(data_T, col, VEN_LIM, sort_col='Freq'):
    freq = pd.DataFrame(columns=[col], data=data_T[col].value_counts()).reset_index()
    freq = freq.rename(columns={'index':col,col:'Freq'})

    data_T_grouped = data_T.groupby(col).sum()
    data_T_grouped = data_T_grouped.reset_index()

    data_T_grouped = data_T_grouped.merge(freq, on=col)
    data_T_grouped = data_T_grouped.sort_values(by=sort_col, ascending=False).head(VEN_LIM)

    return data_T_grouped


def calc_min_dist(df, cols):
    dist = df[:]
    dist['Total']=np.linalg.norm(dist[cols],axis=1)
    dist = dist.sort_values(by=['Total'], ascending=True)
    min_dist = min(dist['Total'])
    
    return [dist,hood]

        
def plot_markers_res(neighborhoods, map_n, c, my_hood, zoom):
    
    size=neighborhoods['Total'].values/min(neighborhoods['Total'])
    # add markers to map
    for s, lat, lng, borough, neighborhood in zip(size,neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
        cc = c
        label = '{}, {}'.format(neighborhood, borough)
        label = folium.Popup(label, parse_html=True)
        if neighborhood==my_hood:
            cc='lime'
        folium.CircleMarker(
            [lat, lng],
            radius=1/s*10,
            popup=label,
            color=cc,
            fill=True,
            fill_color=cc,
            fill_opacity=0.5,
            parse_html=False).add_to(map_n)    
    
    return map_n

def get_top(df, col, num_top):
    
    for hood in df[col]:
        print("----"+ str(hood) +"----")
        temp = df[df[col] == hood].T.reset_index()
        temp.columns = ['venue','score']
        temp = temp.iloc[1:]
        temp['score'] = temp['score'].astype(float)
        temp = temp.round({'score': 2})
        print(temp.sort_values('score', ascending=False).reset_index(drop=True).head(num_top))
        print('\n')
        
def return_most_common_venues(row, num_top_venues):
    
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

# create dataframe for to venues in each hood
def top_venues_df(df, num_top_venues, columns, col):

    indicators = ['st', 'nd', 'rd']  
    for ind in np.arange(num_top_venues):
        try:
            columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
        except:
            columns.append('{}th Most Common Venue'.format(ind+1))

    # create a new dataframe
    neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
    neighborhoods_venues_sorted[col] = df[col]

    for ind in np.arange(df.shape[0]):
        neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(df.iloc[ind, :], num_top_venues)
    
    return neighborhoods_venues_sorted


