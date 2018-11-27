import os
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

from code_felix.car.distance_reduce import getDistance
from code_felix.car.holiday import get_holiday
from code_felix.utils_.util_log import *

from code_felix.utils_.util_cache_file import *

from code_felix.utils_.util_cache_file import *
from code_felix.utils_.util_pandas import *


from functools import lru_cache

DATA_DIR = './input'




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

mini_list = ['861661609024711','2016061820000b' ]

#@lru_cache()
@file_cache(overwrite=False)
def get_train_with_distance(train_file):
    from code_felix.car.distance_reduce import getDistance
    train = get_time_extend(train_file)
    train = get_geo_extend(train)
    train['label'] = 'train'
    train['distance'] = train.apply(lambda row: getDistance(row.start_lat, row.start_lon, row.end_lat, row.end_lon), axis=1)
    return train

@timed()
def get_out_id_attr(file):
    train = get_train_with_distance(file)
    gp = train.groupby('out_id', as_index=False).agg({'distance':['min', 'max', 'mean']})
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
    gp = df.groupby(['out_id', 'end_zoneid'],as_index=False  ).agg({'last_date':['min', 'max', 'count']})
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


@timed()
def get_time_extend(file):

    try:
        df = pd.read_csv(file, delimiter=',' , parse_dates=['start_time', 'end_time'], dtype=train_dict)
    except:
        df = pd.read_csv(file, delimiter=',', parse_dates=['start_time'],   dtype=test_dict)

    #df = df[df.out_id=='2016061820000b']

    df.out_id = df.out_id.astype('str')
    #df = df[df.out_id.isin(mini_list) ]

    df = round(df, 5)
    df['start_base'] = df.start_time.dt.date
    df['holiday'] = df.start_time.dt.date.apply(lambda val: get_holiday(val))
    df['day'] = df.start_time.dt.day
    df['weekday'] = df.start_time.dt.weekday
    df['weekend'] = df.weekday // 5
    df['hour'] = round(df.start_time.dt.hour +df.start_time.dt.minute/60, 2)

    from code_felix.utils_.util_date import distance_2_monday
    df['hour_wk'] = df.apply(lambda row: distance_2_monday(row.start_time), axis=1)

    if 'end_time' in df:
        df['duration'] = (df['end_time'] - df['start_time']) / np.timedelta64(1, 'D')
    else:
        df['duration'] = None

    return df



def adjust_position_2_center(threshold, df, train_file):
    df.out_id = df.out_id.astype(str)

    # logger.debug(df[df.r_key=='SDK-XJ_78d749a376e190685716a51a6704010b'].values)
    from code_felix.car.distance_reduce import reduce_address
    zoneid = reduce_address(threshold, train_file)
    zoneid.out_id = zoneid.out_id.astype(str)


    zoneid = zoneid[['out_id', 'lat', 'lon', 'center_lat', 'center_lon', 'zoneid','sn']]
    zoneid.columns = ['out_id', 'start_lat', 'start_lon', 'start_lat_adj', 'start_lon_adj', 'start_zoneid', 'start_sn']

    #logger.debug(zoneid[zoneid.out_id=='2016061820000b'].values)

    # all = pd.merge(df, zoneid, how='left', on=['out_id', 'start_lat', 'start_lon'] )
    # check_exception(all ,'r_key')
    #
    # all.start_zoneid = all.start_zoneid.astype(int)

    if 'end_time' in df:
        zoneid.columns = ['out_id', 'end_lat', 'end_lon', 'end_lat_adj', 'end_lon_adj', 'end_zoneid','end_sn']
        df = pd.merge(df, zoneid, how='left',on=['out_id', 'end_lat', 'end_lon'] )
        check_exception(df, 'r_key')
        df.end_zoneid = df.end_zoneid.astype(int)

    return df


@file_cache(overwrite=False)
def get_train_with_adjust_position(threshold, train_file):
    train = get_train_with_distance(train_file)
    all = adjust_position_2_center(threshold, train, train_file)

    all = fill_out_id_attr( train_file, all,)
    # all = all.set_index('r_key')
    #all.drop(['index'], axis=1, inplace=True)

    # all = cal_distance_2_centers(all, train_file, threshold, 4)
    return all

@timed()
def analysis_start_zone_id(threshold, train_file, df=None):
    train = get_train_with_adjust_position(threshold, train_file)
    start_zoneid = train.groupby(['out_id', 'start_zoneid', ]).agg(
        {'duration': ['min', 'max', 'mean', 'sum'],
         'distance': ['min', 'max', 'mean', 'sum'],
         'end_sn':['min', 'max', 'mean', 'sum'],

         'end_zoneid': ['nunique', 'count'],
         'start_sn': 'max',
         })



    start_zoneid = flat_columns(start_zoneid, 'sz')
    # logger.debug(f'======={start_zoneid.columns}')

    start_zoneid = start_zoneid.reset_index()

    #check_exception(start_zoneid)

    df = train if df is None else df

    logger.debug(f"Begin to join zoneid#{len(start_zoneid)} info to DF#{len(df)}")
    train_with_zoneid = pd.merge(df, start_zoneid, on=['out_id', 'start_zoneid'], how='left')
    logger.debug(f"End to join zoneid info to DF{len(train_with_zoneid)}")

    for col in train_with_zoneid:
        if 'sz_' in col:
            logger.debug(f'Try to fillna for col#{col}:{round(train_with_zoneid[col].mean(),5)}')
            train_with_zoneid[col].fillna(train_with_zoneid[col].mean(), inplace=True)

    return train_with_zoneid


@timed()
def cal_distance_2_centers(add_with_zoneid, train_file, threshold, topn):
    from code_felix.car.distance_reduce import getDistance
    center_add = get_centers_add(train_file, threshold, topn)
    logger.debug(center_add.columns)
    logger.debug(add_with_zoneid.columns)
    add_with_zoneid = pd.merge(add_with_zoneid, center_add, on='out_id', how='left')
    for i in range(0,topn):
        logger.debug("Try to cal distance to center#%s" % i)
        add_with_zoneid['dis_center_%s'%i] = \
            add_with_zoneid.apply(lambda row: getDistance(row.start_lat_adj, row.start_lon_adj,
                                                          row[f'center_lat_{i}'], row[f'center_lon_{i}']), axis=1)

        add_with_zoneid['dis_center_%s' % i] = round(add_with_zoneid['dis_center_%s'%i])


    #In case some zoneid only have several center
    add_with_zoneid[['dis_center_%s' % i for i in range(0, topn)]] = \
            add_with_zoneid[['dis_center_%s' % i for i in range(0, topn)]].fillna( method='ffill', axis=1)

    add_with_zoneid.drop([f'center_lat_{i}' for i in range(0, topn)], axis=1)
    add_with_zoneid.drop([f'center_lon_{i}' for i in range(0, topn)], axis=1)
    return add_with_zoneid


@timed()
def get_centers_add(train_file, threshold, topn):
    from code_felix.car.distance_reduce import count_in_out_4_zone_id, reduce_address
    zoneid = reduce_address(threshold, train_file)
    in_out = count_in_out_4_zone_id(zoneid, train_file)
    in_out = in_out[(in_out.sn <= topn)]

    # gp[['zoneid_n','lat_n', 'lon_n']] = gp.groupby('out_id')['zoneid','center_lat','center_lon'].shift(-1)
    # gp = gp[gp.sn==0]

    # gp['dis_home_company'] = gp.apply(lambda row: getDistance(row.center_lat, row.center_lon, row.lat_n, row.lon_n), axis=1)
    gp = in_out.pivot_table(['center_lat', 'center_lon'], index=['out_id'], columns='sn')
    gp.reset_index(inplace=True)
    gp = flat_columns(gp)
    return gp

@file_cache(overwrite=False)
def get_test_with_adjust_position(threshold, train_file, test_file):

    test = get_time_extend(test_file)

    real_out_id_list = get_score_outid().out_id
    logger.debug(f'There are {len(real_out_id_list)} outid in submission test file')
    test = test[test.out_id.isin(real_out_id_list)]

    if 'end_time' in test:
        test['distance'] = test.apply(lambda row: getDistance(row.start_lat, row.start_lon, row.end_lat, row.end_lon), axis=1)
        del test['end_time']

    all = adjust_position_2_center(threshold, test, train_file)

    all = fill_out_id_attr( train_file, all,)
    # all = all.set_index('r_key')
    # all.drop(['index'], axis=1, inplace=True)
    # all = cal_distance_2_centers(all, train_file, threshold, 4)

    all = get_geo_extend(all)
    # logger.debug(all.head(1))
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

    mini_train = train[train.out_id==out_id].copy()

    #logger.debug(mini.columns)
    mini_train = mini_train[['end_zoneid', 'end_lat_adj', 'end_lon_adj', 'end_sn']].drop_duplicates()
    mini_train = mini_train.set_index('end_zoneid')

    predict_cols = ['predict_lat','predict_lon', 'predict_sn']
    #logger.debug(test.head(5))
    mini_test = pd.concat([test[test.out_id==out_id], pd.DataFrame(columns=predict_cols)])
    mini_test[predict_cols] = mini_train.loc[mini_test.predict_zone_id].values
    if 'end_lat' in mini_test:
        #mini_test['end_zoneid'] =
        pass

    # logger.debug(test.head(1))
    return mini_test


def cal_loss_for_df(df):
    from code_felix.car.distance_reduce import getDistance
    if df is not None and not df.empty and 'end_zoneid' in df:
        logger.debug(df.columns)
        df['loss_dis'] = df.apply(lambda row: getDistance(row.end_lat, row.end_lon, row.predict_lat, row.predict_lon ) , axis=1)
        df['final_loss'] = df.apply(lambda row: loss_fun(row.loss_dis), axis=1)
        final_loss = round(df.final_loss.mean(), 5)
        accuracy = np.mean(df.end_zoneid == df.predict_zone_id)
        return final_loss, round(accuracy,4)
    else:
        #logger.debug(f"Sub model, for car:{df.out_id.values[0]} with {len(df)} records")
        return None, None

@lru_cache()
def get_feature_columns(gp='0'):
    gp = str(gp)
    logger.debug(f'gp_column:{gp}')
    feature_col = ['weekday', 'weekend',  # 'weekday',
                   # 'holiday',
                   'start_lat', 'start_lon',
                   'hour',
                   #'start_zoneid',
                   #'dis_center_0', #'dis_center_1','dis_center_2'
                   ]
    if gp == '0':
        pass
    elif gp.isnumeric():
        gp = int(gp)
        feature_col.extend([f'geo_rd_c{gp}_{i}' for i in range(gp)])
    elif gp == 'knn':
        feature_col.extend(['dis_center_1', 'dis_center_2', ])
    elif gp == 'zoneid':
        feature_col.extend(['start_zoneid'])
    else:
        logger.warning(f"Can not find gp#{gp} in fun#get_feature_columns")

    logger.debug(f'The training feature gp:{gp}, feature_col:{feature_col}')

    return feature_col


def get_score_outid():
    df = get_time_extend('./input/test_new.csv').groupby('out_id').out_id.count()
    df.name = 'count'
    df = df.to_frame().reset_index()
    return df.sort_values('count', ascending=False)

# def clean_train_useless(df):
#     df['last_time'] = df.groupby(['out_id', 'start_zoneid', 'end_zoneid'])['start_time'].transform('max')
#     df['times'] = df.groupby(['out_id', 'start_zoneid', 'end_zoneid'])['out_id'].transform('count')
#
#     mini = df[df.last_time <= pd.to_datetime('2018-05-01')]
#     mini = mini[mini.times <= 3]
#     return df[~df.index.isin(mini.index)]


def save_df(val, sub, ensemble_test, ensemble_train,  path):
    import os
    path = replace_invalid_filename_char(path)
    logger.debug("Save ensmeble result to file:%s" % path)
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    if not val.empty and 'distance' not in val:
        val['distance'] = val.apply(lambda row: getDistance(row.start_lat, row.start_lon, row.end_lat, row.end_lon),
                                      axis=1)
    if not val.empty:
        val.drop([ 'center_lat_0', 'center_lat_1', 'center_lat_2', 'center_lat_3', 'center_lat_4', 'center_lon_0', 'center_lon_1', 'center_lon_2', 'center_lon_3', 'center_lon_4',
                 ], axis=1, inplace=True, errors='ignore')
        val.to_hdf(path, 'val', index=True,)

    sub.drop([ 'center_lat_0', 'center_lat_1', 'center_lat_2', 'center_lat_3', 'center_lat_4', 'center_lon_0', 'center_lon_1', 'center_lon_2', 'center_lon_3', 'center_lon_4',
                 ], axis=1, inplace=True, errors='ignore')

    sub.to_hdf(path, 'sub', index=True, )

    ensemble_test.to_hdf(path, 'ensemble_test', index=True, )
    ensemble_train.to_hdf(path, 'ensemble_train', index=True, )
    return path


def save_result_partition(val, sub, path):
    if val is not None:
        train = get_time_extend(train_file)
        train = train[['r_key', 'out_id']]
        val = pd.merge(val, train, on='r_key', how='left')
        val.to_hdf(path, 'val', index=True, )

    if sub is not None:
        test = get_time_extend(test_file)
        test = test[['r_key', 'out_id']]
        if 'r_key' not in sub:
            sub = sub.reset_index()
        sub = pd.merge(sub, test, on='r_key', how='left')
        sub.to_hdf(path, 'sub', index=True, )

    logger.debug(f"Partition result save to path:{path}")
    return path

@timed()
def get_geo_extend(df):
    import Geohash as geo
    #geo.encode(42.6, -5.6, precision=6)

    if 'end_lat' in df:
        for i in [4, 5, 6]:
            df[f'geo{i}_cat_end'] = \
                df.apply(lambda row: geo.encode(float(row.end_lat), float(row.end_lon), precision=i), axis=1)
            df[f'geo{i}_end'] = pd.Categorical(df[f'geo{i}_cat_end']).codes

    for i in [4, 5, 6]:
        df[f'geo{i}_cat'] = \
            df.apply(lambda row: geo.encode(float(row.start_lat), float(row.start_lon), precision=i), axis=1)
        df[f'geo{i}'] = pd.Categorical(df[f'geo{i}_cat']).codes


    return df


def reduce_geo(n_components=10, precision=5, ):
    level = f'geo{precision}_cat'

    logger.debug(f'SVD base on {train_file}')
    train = get_train_with_distance(train_file)

    test = get_test_with_adjust_position(500, train_file, test_file)

    test_out = test[['out_id', level, 'hour_wk']]
    test_out.hour_wk = 'out_' + test_out.hour_wk.astype(str)

    car_out = train[['out_id', level, 'hour_wk']]
    car_out.hour_wk = 'out_' + car_out.hour_wk.astype(str)

    car_in = train[['out_id', f'{level}_end', 'hour_wk']]
    car_in.hour_wk = 'in_' + car_in.hour_wk.astype(str)
    car_in.columns = ['out_id', level, 'hour_wk']

    car_in_out_raw = pd.concat([test_out, car_in, car_out])

    car_in_out = car_in_out_raw.groupby([level, 'hour_wk']).agg({'out_id': ['count', 'nunique']})

    car_in_out = flat_columns(car_in_out)

    car_in_out = car_in_out.reset_index()

    from sklearn.utils.extmath import randomized_svd

    svd_input = car_in_out.pivot(index=level, columns='hour_wk', values='out_id_count')

    svd_input.fillna(0, inplace=True)

    U, Sigma, VT = randomized_svd(svd_input.values,
                                  n_components=int(n_components),
                                  n_iter=5,
                                  random_state=None)
    columns = [f'geo_rd_c{n_components}_{i}' for i in range(U.shape[1])]
    reduce = pd.DataFrame(U, index=svd_input.index, columns=columns)

    reduce.index.name=level

    return reduce.reset_index()



if __name__ == '__main__':
    df = get_train_with_adjust_position(150)
    logger.debug(df.shape)
    #
    # print(loss_fun(0))
    # print(loss_fun(100))

