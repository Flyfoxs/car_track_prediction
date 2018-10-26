import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from code_felix.utils_.util_cache_file import *
from code_felix.utils_.util_pandas import *

from math import radians, atan, tan, sin, acos, cos


DATA_DIR = './input'
train_file = f'{DATA_DIR}/train.csv'
test_file = f'{DATA_DIR}/test.csv'



@file_cache(overwrite=True)
def get_train_with_distance():
    train = get_time_extend(train_file)
    train['label'] = 'train'
    train['distance'] = train.apply(lambda row: getDistance(row.start_lat, row.start_lon, row.end_lat, row.end_lon), axis=1)
    train['distance_min' ] =   train.groupby('out_id')['distance'].transform( 'min')
    train['distance_max' ] =   train.groupby('out_id')['distance'].transform( 'max')
    train['distance_mean'] =   train.groupby('out_id')['distance'].transform( 'mean')
    return train

#@file_cache()
def get_time_extend(file):
    try:
        df = pd.read_csv(file, delimiter=',' , parse_dates=['start_time', 'end_time'])
    except:
        df = pd.read_csv(file, delimiter=',', parse_dates=['start_time'])
    df.out_id = df.out_id.astype('str')
    df['start_base'] = df.start_time.dt.date
    df['day'] = df.start_time.dt.day
    df['weekday'] = df.start_time.dt.weekday
    df['weekend'] = df.weekday // 5
    df['hour'] = round(df.start_time.dt.hour +df.start_time.dt.minute/60, 2)
    df['duration'] = (df['end_time'] - df['start_time']) / np.timedelta64(1, 'D')

    return df

def adjust_position(threshold):

    train = get_train_with_distance()
    from code_felix.car.distance_reduce import reduce_address
    zoneid = reduce_address(threshold)
    zoneid = zoneid[['out_id','lat','lon', 'center_lat', 'center_lon', 'zoneid']]
    zoneid.columns =  ['out_id', 'start_lat', 'start_lon', 'start_lat_adj', 'start_lon_adj', 'start_zoneid']

    all = pd.merge(train, zoneid, how='left', )

    zoneid.columns = ['out_id', 'end_lat', 'end_lon', 'end_lat_adj', 'end_lon_adj', 'end_zoneid']

    all = pd.merge(all, zoneid, how='left',)
    return all


def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)

    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
        return distance  # meter
    except:
        return 0.0000001


def show_sample_in_map(sample):
    os.environ['PROJ_LIB'] = '/opt/conda/share/proj'
    from mpl_toolkits.basemap import Basemap
    # create new figure, axes instances.
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 2.8, 2.8])

    llcrnrlon = min(sample.start_lon.min(), sample.end_lon.min()) - 1
    urcrnrlon = max(sample.start_lon.max(), sample.end_lon.max()) + 1

    llcrnrlat = min(sample.start_lat.min(), sample.end_lat.min()) - 1
    urcrnrlat = max(sample.start_lat.max(), sample.end_lat.max()) + 1

    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
                rsphere=(6378137.00, 6356752.3142), \
                resolution='l', projection='merc', \
                lat_0=40., lon_0=-20., lat_ts=20.)

    #     m = Basemap(projection='stere',  #stere, merc
    #                   lat_0=35, lon_0=110,
    #                   llcrnrlon=82.33,
    #                   llcrnrlat=3.01,
    #                   urcrnrlon=138.16,
    #                   urcrnrlat=53.123,resolution='l',area_thresh=10000,rsphere=6371200.)

    for _, item in sample.iterrows():
        # nylat, nylon are lat/lon of New York
        start_lat = item.start_lat;
        start_lon = item.start_lon
        # lonlat, lonlon are lat/lon of London.
        end_lat = item.end_lat;
        end_lon = item.end_lon
        # draw great circle route between NY and London
        m.drawgreatcircle(start_lon, start_lat, end_lon, end_lat, linewidth=2, color='b')
    m.drawcoastlines()
    m.fillcontinents()

    # draw parallels
    parallels = np.arange(0., 90, 10.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)  # 绘制纬线

    meridians = np.arange(80., 140., 10.)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)  # 绘制经线

    m.readshapefile('/home/jovyan/map/CHN_adm1', 'states', drawbounds=True)
    # draw meridians
    m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])
    ax.set_title(f'Great Circle from Start to end for {len(sample)} track')
    plt.show()


def get_distance_color(distance, avg):
    if distance<=2*avg:
        return 'blue'
    else:
        return 'red'
    #'Greens', 'Oranges', 'Reds',

def time_gap(t1, t2):
    gap = abs(t1-t2)
    if gap > 12 :
      gap =  24-gap
    return round(gap/24,2)

if __name__ == '__main__':
    df = adjust_position()
    logger.debug(df.shape)