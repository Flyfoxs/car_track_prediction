from code_felix.car.utils import *

@timed()
def cal_distance_gap_lat():
    train = pd.read_csv(train_file, delimiter=',', parse_dates=['start_time'])
    test = pd.read_csv(train_file, delimiter=',', parse_dates=['start_time'])

    df_list = [train[['out_id', 'start_lat', 'start_lon']],
               train[['out_id', 'end_lat', 'end_lon']],
               test[['out_id', 'start_lat', 'start_lon']],
               ]

    for df in df_list:
        df.columns = ['out_id', 'lat', 'lon']

    place_list = pd.concat(df_list)
    old_len = len(place_list)
    place_list = place_list.drop_duplicates()
    logger.debug(f"There are { old_len - len(place_list)} duplicates address drop from {old_len} records")

    place_list = place_list.sort_values(['out_id', 'lat', 'lon']).reset_index(drop=True)

    place_list['lat_2'] = place_list['lat'].shift(1)
    place_list['lon_2'] = place_list['lon'].shift(1)
    place_list['gap'] = np.abs(place_list['lat_2'] - place_list['lat']) + np.abs(
        place_list['lon_2'] - place_list['lon'])
    place_list['distance_gap'] = round(
        place_list.apply(lambda val: getDistance(val.lat_2, val.lon_2, val.lat, val.lon), axis=1))
    return place_list

@timed()
def cal_zoneid(df, distance_threshold):
    gp = df.groupby('out_id')
    gp_list = []
    for index, df in gp:
        gp_list.append(df)
    logger.debug(f"Already split the gp to mini group:{len(gp_list)}")
    from multiprocessing import Pool as ThreadPool
    from functools import partial
    cal_mini_df_ex = partial(cal_mini_df,distance_threshold=distance_threshold )
    pool = ThreadPool(processes=8)
    results = pool.map(cal_mini_df_ex, gp_list)
    pool.close() ; pool.join()

    all = pd.concat(results)
    return all


def cal_mini_df(mini, distance_threshold):
    mini['zoneid'] = None
    mini = mini.reset_index(drop=True)
    for index, item in mini.iterrows():
        if index == 0:
            mini.loc[index, 'zoneid'] = 0
        elif item.distance_gap <= distance_threshold:
            mini.loc[index, 'zoneid'] = mini.loc[index - 1, 'zoneid']
        else:
            mini.loc[index, 'zoneid'] = mini.loc[index - 1, 'zoneid'] + 1
    logger.debug(f'Get {len(mini.zoneid.drop_duplicates())} zoneid from {len(mini)} records for car:{mini.out_id[0]}, and thresold:{distance_threshold}')

    return mini



@timed()
def cal_center_of_zoneid(place_list):
    place_list[['center_lat', 'center_lon']] = place_list.groupby(['out_id','zoneid'])[['lat', 'lon']].transform('mean')

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


@file_cache(overwrite=False)
def reduce_address(threshold):
    # Cal the distance from previous by lat
    distance_gap_lat = cal_distance_gap_lat()
    #Cal center of zoneid base on lat
    distance_zone_id_lat =  cal_zoneid(distance_gap_lat,threshold)
    # Cal posiztion of center on lat
    center_lat = cal_center_of_zoneid(distance_zone_id_lat)



    #Cal the distance from previous by lon
    distance_gap_lon = cal_distance_gap_center_lon(center_lat)
    # Cal center of zoneid base on lon
    distance_zone_id_lon = cal_zoneid(distance_gap_lon,threshold)
    # Cal posiztion of center on lon
    center_lon = cal_center_of_zoneid(distance_zone_id_lon)

    return center_lon



if __name__ == '__main__':
    for threshold in range(50, 500, 50):
        df = reduce_address(threshold)
        logger.debug(df.shape)
        addressid = df[['out_id', 'zoneid']].drop_duplicates()
        logger.debug(f"Only keep {len(addressid)} address with threshold#{threshold}")
