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


train_train_file = f'{DATA_DIR}/train_train.csv'
train_validate_file = f'{DATA_DIR}/train_validate.csv'

train_file =  f'{DATA_DIR}/train_new.csv'
test_file  =  f'{DATA_DIR}/test_new.csv'

train_dict = {'out_id': str,
              'start_lat': str,
              'start_lon': str,
              'end_lat': str,
              'end_lon': str,

              }


test_dict = {'out_id': str,
              'start_lat': str,
              'start_lon': str,
              }


#@lru_cache()
@file_cache(overwrite=False)
def get_train_with_distance(train_file):
    from code_felix.car.distance_reduce import getDistance
    train = get_time_extend(train_file)
    train['label'] = 'train'
    train['distance'] = train.apply(lambda row: getDistance(row.start_lat, row.start_lon, row.end_lat, row.end_lon), axis=1)
    return train

@timed()
def get_out_id_attr(file):
    train = get_train_with_distance(file)
    gp = train.groupby('out_id').agg({'distance':['min', 'max', 'mean']})
    gp = flat_columns(gp)
    return round(gp.reset_index())

@timed()
def fill_out_id_attr(file, df=None, ):
    out_id_attr = get_out_id_attr(file)
    if df is not None :
        logger.debug(f"Fill df with {out_id_attr.columns}")
        return pd.merge(df, out_id_attr, on='out_id', how='left')
    else:
        return out_id_attr
@timed()
def get_end_zoneid_attr():
    df = get_train_with_adjust_position(100,train_file, test_file)
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
        df = pd.read_csv(file, delimiter=',' , parse_dates=['start_time', 'end_time'], dtype=train_dict)
    except:
        df = pd.read_csv(file, delimiter=',', parse_dates=['start_time'],   dtype=test_dict)


    df.out_id = df.out_id.astype('str')
    df = df[df.out_id == '2016061820000b']

    df = round(df, 5)
    df['start_base'] = df.start_time.dt.date
    df['day'] = df.start_time.dt.day
    df['weekday'] = df.start_time.dt.weekday
    df['weekend'] = df.weekday // 5
    df['hour'] = round(df.start_time.dt.hour +df.start_time.dt.minute/60, 2)
    if 'end_time' in df:
        df['duration'] = (df['end_time'] - df['start_time']) / np.timedelta64(1, 'D')
    else:
        df['duration'] = None

    return df



def adjust_position_2_center(threshold, df):
    df.out_id = df.out_id.astype(str)

    from code_felix.car.distance_reduce import reduce_address
    zoneid = reduce_address(threshold)
    zoneid.out_id = zoneid.out_id.astype(str)

    zoneid = zoneid[['out_id', 'lat', 'lon', 'center_lat', 'center_lon', 'zoneid']]
    zoneid.columns = ['out_id', 'start_lat', 'start_lon', 'start_lat_adj', 'start_lon_adj', 'start_zoneid']

    all = pd.merge(df, zoneid, how='left', )
    check_exception(all ,'r_key')

    all.start_zoneid = all.start_zoneid.astype(int)

    if 'end_time' in df:
        zoneid.columns = ['out_id', 'end_lat', 'end_lon', 'end_lat_adj', 'end_lon_adj', 'end_zoneid']
        all = pd.merge(all, zoneid, how='left', )
        check_exception(all, 'r_key')
        all.end_zoneid = all.end_zoneid.astype(int)

    return all


@file_cache(overwrite=False)
def get_train_with_adjust_position(threshold, train_file):
    train = get_train_with_distance(train_file)
    all = adjust_position_2_center(threshold, train)

    all = fill_out_id_attr( train_file, all,)
    all = all.set_index('r_key')
    all.drop(['index'], axis=1, inplace=True)
    return all


@file_cache(overwrite=False)
def get_test_with_adjust_position(threshold, train_file, test_file):

    test = get_time_extend(test_file)
    if 'end_time' in test:
        del test['end_time']

    all = adjust_position_2_center(threshold, test)

    all = fill_out_id_attr( train_file, all,)
    all = all.set_index('r_key')
    all.drop(['index'], axis=1, inplace=True)
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


def get_zone_inf(out_id, train, test):
    mini_train = train[train.out_id==out_id]
    #logger.debug(mini.columns)
    mini_train = mini_train[['end_zoneid', 'end_lat_adj', 'end_lon_adj']].drop_duplicates()
    mini_train = mini_train.sort_values('end_zoneid').reset_index(drop=True)

    predict_cols = ['predict_zone_id', 'predict_lat','predict_lon']
    test = pd.concat([test, pd.DataFrame(columns=predict_cols)])
    test[predict_cols] = mini_train.loc[test.predict_id].values

    return test


def cal_loss_for_df(df):
    from code_felix.car.distance_reduce import getDistance
    if 'end_lat' in df:
        df['loss_dis'] = df.apply(lambda row: getDistance(row.end_lat, row.end_lon, row.predict_lat, row.predict_lon ) , axis=1)
        df['final_loss'] = df.apply(lambda row: loss_fun(row.loss_dis), axis=1)
        final_loss = round(df.final_loss.mean(), 5)
        out_id_len = len(df.out_id.drop_duplicates())
        if out_id_len==1:
            logger.debug(f'loss is {final_loss}, for car:{df.out_id[0]} with {len(df)} records')
        else:
            logger.debug(f'loss for {out_id_len} out_id is {final_loss}')
        return final_loss
    else:
        return None


# def clean_train_useless(df):
#     df['last_time'] = df.groupby(['out_id', 'start_zoneid', 'end_zoneid'])['start_time'].transform('max')
#     df['times'] = df.groupby(['out_id', 'start_zoneid', 'end_zoneid'])['out_id'].transform('count')
#
#     mini = df[df.last_time <= pd.to_datetime('2018-05-01')]
#     mini = mini[mini.times <= 3]
#     return df[~df.index.isin(mini.index)]


if __name__ == '__main__':
    df = get_train_with_adjust_position(150)
    logger.debug(df.shape)
    #
    # print(loss_fun(0))
    # print(loss_fun(100))

