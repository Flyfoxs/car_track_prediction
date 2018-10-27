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
    from code_felix.car.distance_reduce import getDistance
    train = get_time_extend(train_file)
    train['label'] = 'train'
    train['distance'] = train.apply(lambda row: getDistance(row.start_lat, row.start_lon, row.end_lat, row.end_lon), axis=1)
    return train

@timed()
def get_out_id_attr():
    train = get_train_with_distance()
    gp = train.groupby('out_id').agg({'distance':['min', 'max', 'mean']})
    gp = flat_columns(gp)
    return round(gp.reset_index())

def fill_out_id_attr(df=None):
    out_id_attr = get_out_id_attr()
    if df is not None :
        logger.debug(f"Fill df with {out_id_attr.columns}")
        return pd.merge(df, out_id_attr, on='out_id', how='left')
    else:
        return out_id_attr
@timed()
def get_end_zoneid_attr():
    df = get_train_with_adjust_position(100)
    gp = df.groupby(['out_id', 'end_zoneid']).agg({'last_date':['min', 'max', 'count']})
    gp = flat_columns(gp)
    return gp

@timed()
def fill_end_zone_attr(df=None):
    end_zoneid = get_end_zoneid_attr()
    if df is not None :
        logger.debug(f"Fill df with {end_zoneid.columns}")
        return pd.merge(df, end_zoneid, on=['out_id', 'end_zoneid'], how='left')
    else:
        return end_zoneid


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


def loss_fun(gap):
    import math
    return round(1/(1+math.exp((1000-gap)/250)), 5)

if __name__ == '__main__':
    df = get_train_with_adjust_position(100)
    logger.debug(df.shape)
    #
    # print(loss_fun(0))
    # print(loss_fun(100))

