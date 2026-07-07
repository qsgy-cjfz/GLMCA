# -*- coding: utf-8
"""
@author：67543
@date：  2022/3/20
@contact：675435108@qq.com
"""
import math
import random

from data import Data
import pandas as pd
import folium
import plotly
import time


def process_data(data):
    checkin_data = pd.read_csv(data.check_in_file, delimiter="\t", names=['uids', 'lids', 'timestamp','lat','lng'], header=None)
    checkin_data = checkin_data.sort_values(by=['uids','timestamp'])
    poi_coo_data = pd.read_csv(data.poi_file,delimiter="\t", names=['lids', 'lat','lng'], header=None)
    count = 0
    for _,(lid,lat,lng) in poi_coo_data.iterrows():
        start_time = time.time()
        checkin_data.loc[checkin_data['lids'] == int(lid),['lat','lng']] = [lat,lng]
        count += 1
        if count%5001 ==0:
            print ("已执行{}次,用时{}s".format(count,time.time()-start_time))
    return checkin_data


def load_path(data_name):
    for dataset in data_name:
        data = Data(dataset)
        checkin_data = process_data(data)
        checkin_data.to_csv("./process/" + str(dataset) + "_checkin_poi.csv", sep='\t', index=False, header=None)


def generate_map():
    data_name = 'Gowalla'
    data_path = "./process/"+data_name+"_checkin_poi.csv"
    checkin_data = pd.read_csv(data_path,sep="\t",names=['uids','lids','timestamp','lat','lng'])
    user_checkin_groups= checkin_data.groupby(by='uids')
    sigle_user_check = user_checkin_groups.get_group(random.randint(0,5628))
    start_user, start_poi, start_timestamp, start_lat, start_lng = sigle_user_check.iloc[0]
    start = 0
    m = folium.Map(location=[start_lat, start_lng], zoom_start=5,width='100%',height='100%',control_scale=False)  # EPSG4326,Simple,EPSG3395
    pois = []
    for row_num,current_dot in sigle_user_check.iterrows():
         lid,lat,lng = str(int(current_dot['lids'])),current_dot['lat'],current_dot['lng']
         if start < len(sigle_user_check)-1:
             next_dot = sigle_user_check.iloc[start + 1]
             dx, dy = next_dot['lat'] - lat, next_dot['lng'] -lng
             angle = -int(math.atan2(dx, dy) * 180 / math.pi)
             arrows_position1 = [lat + dx / 2, lng + dy / 2]
             arrows_position2 = [lat + dx / 3, lng + dy / 3]
             arrows_position3 = [lat + dx / 3, lng + dy / 3]
             if start == 0:
                 folium.Marker([lat,lng], popup=lid,icon=folium.Icon(color='blue', prefix='fa')).add_to(m)
             else:
                 folium.Marker([lat, lng], popup=str(int(lid))).add_to(m)
                 #folium.CircleMarker(location=[lat, lng],radius=5).add_to(m)
             folium.RegularPolygonMarker(arrows_position2,color='green',fill_color='green', number_of_sides=3, radius=5,
                                         rotation=angle).add_to(m)
         else:
             folium.Marker([lat, lng],popup=lid, icon=folium.Icon(color='red', prefix='fa',icon="location-check")).add_to(m)
         start += 1
         pois.append([current_dot['lat'],current_dot['lng']])
    folium.PolyLine(pois, color='green',opacity=0.9).add_to(m)
    m.save('folium_plot.html')


if __name__ == '__main__':
    # load_path(["Gowalla", 'Yelp'])
    # test_direction()
    generate_map()





