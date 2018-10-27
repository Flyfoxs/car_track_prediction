import os

import numpy as np
import matplotlib.pyplot as plt
from code_felix.utils_.util_log import *

from code_felix.utils_.util_cache_file import *

from code_felix.utils_.util_cache_file import *
from code_felix.utils_.util_pandas import *

from math import radians, atan, tan, sin, acos, cos
from functools import lru_cache

DATA_DIR = './input'
train_file = f'{DATA_DIR}/train.csv'
test_file = f'{DATA_DIR}/test.csv'



@lru_cache()
@file_cache(overwrite=True)
def get_train_with_distance():
    train = get_time_extend(train_file)
    train['label'] = 'train'
    train['distance'] = train.apply(lambda row: getDistance(row.start_lat, row.start_lon, row.end_lat, row.end_lon), axis=1)
    return train

@timed()
def get_out_id_attr():
    train = get_train_with_distance()
    gp = train.groupby('out_id').agg({'distance':['min', 'max', 'mean']})
    gp.columns = ['_'.join(item) for item in gp.columns]
    return round(gp.reset_index())

def fill_out_id_attr(df=None):
    out_id_attr = get_out_id_attr()
    if df is not None :
        logger.debug(f"Fill df with {out_id_attr.columns}")
        return pd.merge(df, out_id_attr, on='out_id', how='left')
    else:
        return out_id_attr



def get_import_address():
    train = get_train_with_adjust_position()


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

@file_cache(overwrite=True)
def get_train_with_adjust_position(threshold):
    train = get_train_with_distance()
    from code_felix.car.distance_reduce import  reduce_address
    zoneid = reduce_address(threshold)
    zoneid = zoneid[['out_id','lat','lon', 'center_lat', 'center_lon', 'zoneid']]
    zoneid.columns =  ['out_id', 'start_lat', 'start_lon', 'start_lat_adj', 'start_lon_adj', 'start_zoneid']

    all = pd.merge(train, zoneid, how='left', )

    zoneid.columns = ['out_id', 'end_lat', 'end_lon', 'end_lat_adj', 'end_lon_adj', 'end_zoneid']

    all = pd.merge(all, zoneid, how='left',)
    all = fill_out_id_attr(all)
    return all

@file_cache()
def get_test_with_adjust_position(threshold):

    test = get_time_extend(test_file)
    from code_felix.car.distance_reduce import reduce_address
    zoneid = reduce_address(threshold)
    zoneid = zoneid[['out_id','lat','lon', 'center_lat', 'center_lon', 'zoneid']]
    zoneid.columns =  ['out_id', 'start_lat', 'start_lon', 'start_lat_adj', 'start_lon_adj', 'start_zoneid']

    all = pd.merge(test, zoneid, how='left', )
    all = fill_out_id_attr(all)
    return all



from code_felix.car.distance_reduce import get_center_address, getDistance
from code_felix.car.utils import *




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
    df = get_train_with_adjust_position(100)
    logger.debug(df.shape)

