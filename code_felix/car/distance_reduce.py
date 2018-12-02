from sklearn.cluster import KMeans

from code_felix.car.utils import *

from code_felix.utils_.util_log import *
from code_felix.utils_.util_date import *
from code_felix.utils_.util_cache_file import *
from multiprocessing import Pool as ThreadPool
from functools import partial

test_columns = ['r_key','out_id','start_time','start_lat','start_lon']

min_gap =500
max_gap =5000



@file_cache()
def cal_distance_gap_and_zoneid(train, test, threshold):
    sorted_address = sort_address_and_cal_gap(train, test)
    address_with_zoneid = cal_zoneid_base_on_gap(sorted_address, threshold)
    return address_with_zoneid

@timed()
def sort_address_and_cal_gap(train_file, test_file):
    from code_felix.car.utils import train_dict, test_dict
    train = pd.read_csv(train_file, delimiter=',', parse_dates=['start_time'], dtype=train_dict)
   # test = pd.read_csv(test_file, usecols=test_columns, delimiter=',', parse_dates=['start_time'], dtype=test_dict)

    #train = train[train.out_id=='2016061820000b']
    #test = test[test.out_id == '2016061820000b']


    df_list = [#train[['out_id', 'start_lat', 'start_lon']],
               train[['out_id', 'end_lat', 'end_lon']],
               #test[['out_id', 'start_lat', 'start_lon']],
               ]

    for df in df_list:
        df.columns = ['out_id', 'lat', 'lon']

    place_list = pd.concat(df_list)
    # place_list = place_list[place_list.out_id.isin(mini_list)]

    old_len = len(place_list)

    place_list.out_id = place_list.out_id.astype(str)

    # place_list = round(place_list, 5)
    place_list = place_list.drop_duplicates()

    logger.debug(f"There are { old_len - len(place_list)} duplicates address drop from {old_len} records")

    place_list = place_list.sort_values(['out_id', 'lat', 'lon']).reset_index(drop=True)

    place_list['lat_2'] = place_list['lat'].shift(1)
    place_list['lon_2'] = place_list['lon'].shift(1)
    # place_list['gap'] = np.abs(place_list['lat_2'] - place_list['lat']) + np.abs(
    #     place_list['lon_2'] - place_list['lon'])
    place_list['distance_gap'] = round(
        place_list.apply(lambda val: getDistance(val.lat_2, val.lon_2, val.lat, val.lon), axis=1))

    from code_felix.car.utils import fill_out_id_attr
    place_list = fill_out_id_attr(train_file, place_list)
    return place_list

@timed()
def cal_zoneid_base_on_gap(df, distance_threshold):
    gp = df.groupby(['out_id'])
    gp_list = []
    for index, df in gp:
        gp_list.append(df)
    logger.debug(f"Already split the gp to mini group:{len(gp_list)}")

    cal_mini_df_ex = partial(cal_mini_df,distance_threshold=distance_threshold )
    pool = ThreadPool(processes=8)
    results = pool.map(cal_mini_df_ex, gp_list)
    pool.close() ; pool.join()

    all = pd.concat(results)
    return all


def cal_mini_df(mini, distance_threshold):
    mini['zoneid'] = None
    mini = mini.reset_index(drop=True)
    #logger.debug(mini.columns)
    if distance_threshold is None or distance_threshold < 100:
        distance_mean = mini.distance_mean.values[0]
        distance_threshold = distance_mean // distance_threshold

        distance_threshold = max(min_gap,distance_threshold)
        distance_threshold = min(max_gap,distance_threshold)


    for index, item in mini.iterrows():
        if index == 0:
            mini.loc[index, 'zoneid'] = 100
        elif item.distance_gap <= distance_threshold:
            mini.loc[index, 'zoneid'] = mini.loc[index - 1, 'zoneid']
        else:
            mini.loc[index, 'zoneid'] = mini.loc[index - 1, 'zoneid'] + 1
    logger.debug(f'Get {len(mini.zoneid.drop_duplicates())} zoneid from {len(mini)} records for car:{mini.out_id[0]}, and thresold:{distance_threshold}')

    return mini



@timed()
def cal_center_of_zoneid(place_list):
    place_list['lat_f'] = place_list.lat.astype(float)
    place_list['lon_f'] = place_list.lon.astype(float)
    place_list[['center_lat', 'center_lon']] = place_list.groupby(['out_id','zoneid'])[['lat_f', 'lon_f']].transform('mean')

    place_list['distance_2_center'] = round(
        place_list.apply(lambda val: getDistance(val.center_lat, val.center_lon, val.lat, val.lon), axis=1))

    return place_list

@timed()
def cal_distance_gap_center_lon(place_list):

    place_list = place_list.sort_values(['out_id', 'center_lon', 'center_lat']).reset_index(drop=True)

    place_list['lat_3'] = place_list['center_lat'].shift(1)
    place_list['lon_3'] = place_list['center_lon'].shift(1)

    place_list['distance_gap'] = round(
        place_list.apply(lambda val: getDistance(val.lat_3, val.lon_3, val.center_lat, val.center_lon), axis=1))
    return place_list

from functools import lru_cache
@lru_cache()
def get_center_address(threshold, train, test):
    df = reduce_address(threshold, train, test)
    mini = df.drop_duplicates(['out_id', 'zoneid', 'center_lat', 'center_lon',])
    return mini

def reduce_address_knn(dis_with_zoneid, threshold=2):
    outid_zoneid_count = dis_with_zoneid.groupby('out_id')['zoneid'].nunique()
    outid_zoneid_count = outid_zoneid_count[outid_zoneid_count>=50]

    total = len(outid_zoneid_count)
    i =0
    for out_id, count in outid_zoneid_count.iteritems():
        i += 1
        from sklearn import neighbors
        logger.debug(f'Try to reduce {count} address by knn : for {out_id}  {i}/{total}')
        n_neighbors = 1
        weights = 'distance'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

        mini = dis_with_zoneid.loc[dis_with_zoneid.out_id==out_id]
        old_len = len(mini.zoneid.drop_duplicates())

        gp_mini = mini.drop_duplicates('zoneid')
        gp_mini = gp_mini.groupby('zoneid')['in_out'].sum().sort_values(ascending=False)

        train = mini[mini.zoneid.isin(gp_mini[gp_mini >= threshold].index)]
        test = mini[mini.zoneid.isin(gp_mini[gp_mini < threshold].index)]
        logger.debug(f'train.shape:{train.shape}, test.shape:{test.shape}')

        X = train[['lat', 'lon']].values
        Y = train.zoneid.astype(int).values

        clf.fit(X, Y)

        zoneid_predict = clf.predict(test[['lat', 'lon']].values)

        dis_with_zoneid.loc[test.index, 'zoneid'] = zoneid_predict

        new_len = len(dis_with_zoneid.loc[dis_with_zoneid.out_id==out_id]['zoneid'].drop_duplicates())

        logger.debug(f'zoneid from reduce from{old_len} to{new_len} for outid:{out_id}')

    dis_with_zoneid = cal_center_of_zoneid(dis_with_zoneid)
    return dis_with_zoneid


def reduce_address_kmeans(dis_with_zoneid, num_center):
    #
    outid_zoneid_count = dis_with_zoneid.groupby('out_id')['zoneid'].nunique()
    outid_zoneid_count = outid_zoneid_count[outid_zoneid_count>=num_center]

    total = len(outid_zoneid_count)
    i =0
    for out_id in outid_zoneid_count.index:
        i += 1
        logger.debug(f'Try to reduce address to:{num_center} for {out_id}  {i}/{total}')
        X = dis_with_zoneid.loc[dis_with_zoneid.out_id == out_id][['lat','lon']]
        kmeans = KMeans(n_clusters=num_center, random_state=777).fit(X)
        center = kmeans.cluster_centers_
        y_pred = kmeans.predict(X)
        dis_with_zoneid.loc[dis_with_zoneid.out_id == out_id, 'zoneid']  = y_pred
        dis_with_zoneid.loc[dis_with_zoneid.out_id == out_id, 'center_lat'] = center[y_pred][:,0]
        dis_with_zoneid.loc[dis_with_zoneid.out_id == out_id, 'center_lon'] = center[y_pred][:, 1]

    return dis_with_zoneid





@file_cache(overwrite=False)
def reduce_address(threshold, train_file):

    test_file = train_file.replace('train_', 'test_')

    #Cal center of zoneid base on lat
    dis_with_zoneid =  cal_distance_gap_and_zoneid(train_file, test_file, threshold)

    # Cal posiztion of center on lat
    dis_with_zoneid = adjust_add_with_centers(dis_with_zoneid, threshold)
    #dis_with_zoneid = adjust_add_with_centers(dis_with_zoneid, threshold)

    freq = count_in_out_4_zone_id(dis_with_zoneid, train_file)
    # freq = freq[['out_id', 'zoneid', 'sn']].drop_duplicates()

    dis_with_zoneid = pd.merge(dis_with_zoneid, freq, how='left')

    logger.debug(f'dis_with_zoneid columns:{dis_with_zoneid.columns}')
    ## 'lat_2', 'lon_2',  'distance_gap' , 'lat_f', 'lon_f', 'zoneid_new', 'zoneid_raw'
    dis_with_zoneid = dis_with_zoneid[['out_id', 'lat', 'lon',
            'zoneid','sn', 'center_lat', 'center_lon',
            'distance_2_center',
               'out', 'in' ,'in_out',
             'in_total', 'out_total', 'in_out_total'	,
             'in_per',	'out_per',	'in_out_per' ,
                                       ]]
    # dis_with_zoneid = reduce_address_kmeans(dis_with_zoneid, top)

    #dis_with_zoneid  = reduce_address_knn(dis_with_zoneid,2)
    return dis_with_zoneid

def reorder_zoneid_frequency(dis_with_zone_id, train_file):
    freq = count_in_out_4_zone_id(dis_with_zone_id, train_file)
    # freq = freq[['out_id', 'zoneid', 'sn']].drop_duplicates()

    dis_with_zone_id = pd.merge(dis_with_zone_id,freq, how='left')
    dis_with_zone_id.zoneid = dis_with_zone_id.apply(lambda row: row.sn if pd.notnull(row.sn) else 9000+row.zoneid, axis=1)

    # dis_with_zone_id.zoneid, dis_with_zone_id.sn =  dis_with_zone_id.sn, dis_with_zone_id.zoneid
    #
    # dis_with_zone_id.zoneid.fillna('999',inplace=True)
    return dis_with_zone_id


def stand_zonid_by_end():
    #TODO
    pass

def count_in_out_4_zone_id(dis_with_zone_id, train_file):
    from code_felix.car.utils import get_train_with_distance
    train = get_train_with_distance(train_file)
    come_out = pd.merge(train, dis_with_zone_id, left_on=['out_id', 'start_lat', 'start_lon'],
                        right_on=['out_id', 'lat', 'lon'], how='left')

    come_out['out'] = 1

    come_in = pd.merge(train, dis_with_zone_id, left_on=['out_id', 'end_lat', 'end_lon'],
                       right_on=['out_id', 'lat', 'lon'], how='left')
    come_in['in'] = 1

    all = pd.concat([come_out, come_in])
    gp = all.groupby(['out_id', 'zoneid', 'center_lat', 'center_lon',]).agg({'out': 'sum', 'in': 'sum'})

    gp['in_out'] = gp.sum(axis=1)

    gp[['in_total', 'out_total', 'in_out_total']] = gp.groupby('out_id')['in','out','in_out'].transform('sum')

    gp['in_per'] =  gp['in']/gp.in_total
    gp['out_per'] = gp['out']/gp.out_total
    gp['in_out_per'] = gp['in_out']/gp.in_out_total
    #gp.reset_index().sort_values(['out_id', 'in_out'], ascending=False)

    gp = gp.reset_index().sort_values(['out_id', 'in_out'], ascending=False)
    gp['sn'] = gp.groupby(['out_id'])['zoneid'].cumcount()
    gp['previous'] = gp.groupby('out_id')['in_out'].shift(1)
    gp['drop_percent'] = gp.in_out / gp.previous
    gp.drop_percent.fillna(1, inplace=True)
    return gp



#Far from the home&company, and only 1 or 2 times
def pickup_stable_zoneid(adress_with_zoneid):
    pass

# def reduce_center_address(center_lat,threshold):
#     mini = center_lat.drop_duplicates(['out_id', 'zoneid', 'center_lat', 'center_lon', ])
#     mini = mini.reset_index(drop=True)
#     #0.00001 neeary 1 meter in map
#     lon_threshold = 0.00001*threshold
#     merge_list = []
#     for index, cur in mini.iterrows():
#         for next_i, next in mini[index+1:]:
#             if cur.out_id != next.out_id:
#                 break
#             elif abs(cur.center_lon - next.center_lon) > lon_threshold:
#                 break
#             elif lon_threshold > getDistance(cur.center_lat.values, cur.center_lon.values, next.center_lat.values, next.center_lon.values, ):
#                 merge_item = (next.out_id.values, next.zoneid.values, cur.out_id.values, cur.zoneid.values)
#                 logger.debug('Merge %s#%s to %s#%s ' % merge_item)
#                 merge_list.append(merge_item)
#             else:
#                 pass
#     return merge_list




def getDistance(latA, lonA, latB, lonB):
    from math import radians, atan, tan, sin, acos, cos
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(float(latA))
    radLonA = radians(float(lonA))
    radLatB = radians(float(latB))
    radLonB = radians(float(lonB))

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

def get_distance_zoneid(threshold, out_id, id1, id2 , train, test):
    """
    get_distance_zoneid(100, '358962079107966', 89, 105)
    :param threshold:
    :param out_id:
    :param id1:
    :param id2:
    :return:
    """
    df = get_center_address(threshold, train, test)
    add1 = df[(df.out_id==out_id) & (df.zoneid==id1)]
    add2 = df[(df.out_id==out_id) & (df.zoneid==id2)]
    #print(add1.center_lat, add1.center_lon, add2.center_lat, add2.center_lon, '==')
    #print("====":add1.center_lat.values, "====")
    dis = getDistance(add1.center_lat.values, add1.center_lon.values, add2.center_lat.values, add2.center_lon.values,)
    logger.debug(f'Distance between zoneids({id1} and {id2}) is <<{round(dis)}>> for car#{out_id}')
    return dis

@timed()
def get_center_address_need_reduce(dis_with_zoneid,threshold):
    center_lat = cal_center_of_zoneid(dis_with_zoneid)
    mini = center_lat.drop_duplicates(['out_id', 'zoneid', 'center_lat', 'center_lon', ])
    mini  = mini[mini.zoneid >=100]

    out_id_list = mini.out_id.drop_duplicates()
    logger.debug(f"There are {len(out_id_list)} out_id need to reduce ")
    out_id_split_list = []
    for out_id in out_id_list:
        out_id_mini = mini.loc[mini.out_id==out_id,:]
        out_id_mini = out_id_mini.sort_values([ 'center_lon', 'center_lat'])
        out_id_mini = out_id_mini.reset_index(drop=True)

        out_id_split_list.append(out_id_mini)

    cal_reduce_center_add = partial(get_center_address_need_reduce_for_one_out_id, threshold= threshold)

    pool = ThreadPool(processes=8)
    results = pool.map(cal_reduce_center_add, out_id_split_list)
    pool.close();
    pool.join()

    df = pd.concat(results)

    logger.debug(f"There are {len(df)} zoneid need to merge")
    return df


def get_center_address_need_reduce_for_one_out_id(out_id_mini,threshold ):
    if threshold is None or threshold < 100:
        distance_mean = out_id_mini.distance_mean.values[0]
        threshold = distance_mean // threshold
        threshold = max(min_gap,threshold)
        threshold = min(max_gap,threshold)
    #logger.debug(f'centerid need to reduce, out_id:{out_id_mini.at[0,"out_id"]}, threshold:{threshold}')


    lon_threshold = 0.00001 * threshold * 2
    df = pd.DataFrame(columns=['out_id', 'zoneid', 'zoneid_new', 'cur_dis',])
    zoneid_replaced = []
    for index, cur in out_id_mini.iterrows():
        #logger.debug(f"=======Cur:{cur.zoneid}@out_id:{cur.out_id}")
        compare = out_id_mini[index + 1:]
        for next_i, next_ in compare.iterrows():
            # logger.debug(f'{index}/{next_i}')
            cur_dis = round(getDistance(cur.center_lat, cur.center_lon, next_.center_lat, next_.center_lon, ), )
            cur_lon_threshold = round(abs(cur.center_lon - next_.center_lon), 5)
            if cur_lon_threshold > lon_threshold:
                # logger.debug(f'cur_lon {cur_lon_threshold}:{cur.center_lon}, {next_.center_lon}')
                # logger.debug(f'cur_lon {cur_lon_threshold}, cur/next:{cur.zoneid}/{next_.zoneid}, dis:{cur_dis}, over the lon threshold#{lon_threshold}')
                break
            elif next_.zoneid in zoneid_replaced or cur.zoneid in zoneid_replaced:
                # logger.debug(f'{next_.zoneid} or {cur.zoneid} already in list')
                pass
            elif cur_dis <= threshold:
                zoneid_replaced.append(next_.zoneid)
                merge_item = (next_.out_id, next_.zoneid, cur.zoneid, cur_dis,)
                df.loc[len(df)] = list(merge_item)
                #logger.debug('Merge %s#%s to %s, distance_gap:%s ' % merge_item)
            else:
                pass
    return df


@timed()
def adjust_add_with_centers(address_list, threshold):
    old_len = len(address_list.drop_duplicates(['out_id', 'zoneid']))
    reduce_list = get_center_address_need_reduce(address_list, threshold)
    #logger.debug(f'The reduce list is :{len(reduce_list)}, \n {reduce_list[:10]}')
    reduce_list = reduce_list[['out_id', 'zoneid', 'zoneid_new']]

    address_list = pd.merge(address_list,reduce_list, how='left')
    if 'zoneid_raw' not in address_list:
        address_list['zoneid_raw'] = address_list['zoneid']

    address_list['zoneid'] = np.where(pd.isna(address_list.zoneid_new), address_list.zoneid, address_list.zoneid_new)

    new_len = len(address_list.drop_duplicates(['out_id', 'zoneid']))
    logger.debug(f"There are {old_len- new_len} zone is removed, current zone is {new_len},original is {old_len}")
    return cal_center_of_zoneid(address_list)

def get_home_company():

    train = get_train_with_adjust_position(200, train_file, test_file)


@lru_cache()
#@file_cache()
def get_outid_geo_summary():
    from code_felix.ensemble.stacking import train_file, test_file
    from code_felix.car.utils import get_time_geo_extend
    train_geo = get_time_geo_extend(train_file)
    geo_sum = train_geo.groupby('out_id').agg({'geo4_cat': 'nunique', 'geo5_cat': 'nunique',
                                               'geo6_cat': 'nunique', 'geo7_cat': 'nunique',
                                               'geo8_cat': 'nunique', 'geo9_cat': 'nunique',
                                               'geo4_cat_end': 'nunique', 'geo5_cat_end': 'nunique',
                                               'geo6_cat_end': 'nunique', 'geo7_cat_end': 'nunique',
                                               'geo8_cat_end': 'nunique', 'geo9_cat_end': 'nunique',
                                               'r_key': 'count'})
    geo_sum.rename(columns={'r_key': 'count'}, inplace=True, )
    geo_sum['gp'] = pd.cut(geo_sum.geo6_cat_end, [0, 20, 40, 60, 80, 100, 900]).cat.codes
    geo_sum.columns = [f'train_{col}' for col in geo_sum.columns]

    test_geo = get_time_geo_extend(test_file)
    test_geo_sum = test_geo.groupby('out_id').agg({'geo4_cat': 'nunique', 'geo5_cat': 'nunique',
                                                   'geo6_cat': 'nunique', 'geo7_cat': 'nunique',
                                                   'geo8_cat': 'nunique', 'geo9_cat': 'nunique',
                                                   'r_key': 'count'})
    test_geo_sum.rename(columns={'r_key': 'count'}, inplace=True, )
    test_geo_sum.columns = [f'test_{col}' for col in test_geo_sum.columns]

    return pd.concat([geo_sum, test_geo_sum], axis=1)


def adjust_zoneid_base_geo6(df):
    df.end_zoneid = df.geo6

    df.end_lat = df.end_lat.astype(float)
    df.end_lon = df.end_lon.astype(float)

    df.end_lat_adj = df.groupby(['out_id','geo6'] )['end_lat'].transform('mean')
    df.end_lon_adj = df.groupby(['out_id','geo6'] )['end_lon'].transform('mean')
    return df


if __name__ == '__main__':
    for threshold in [300,  500, 600,220,400, ]:
        for file in ['train_new.csv', 'train_val_all.csv', ]:
            file = f'./input/{file}'
            # reduce_address(threshold,file)
    # logger.debug(df.shape)
    # addressid = df[['out_id', 'zoneid']].drop_duplicates()
    # logger.debug(f"Only keep {len(addressid)} address with threshold#{threshold}")
    #
    # df = get_train_with_adjust_position(100, train_file)
    # logger.debug(df.shape)
    #
    # #get_distance_zoneid(100, '358962079107966', 72, 73, train_file, test_file)

    pass