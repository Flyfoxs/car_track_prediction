from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
import pandas as pd
import numpy as np

def split_to_group(num=5, base_file='./'):
    """
    'start_zoneid', 'loss_dis', 'final_loss', 'out_id', 'predict_id', 'sn',  'predict_zone_id'

    zoneid, max(loss), min(loss), vag(loss)
    distinance, avg, max
    count(*), count(zoneid) in train, test

    train: count(end_zoneid/start_zoneid)

    :param num:
    :param base_file:
    :return:
    """
    pass

def analysis_train_test(threshold=500):
    train = get_train_with_adjust_position(threshold, './input/train_new.csv')
    train.groupby()

    test = get_train_with_adjust_position(threshold, './input/test_new.csv')


@file_cache()
def get_worse_case(score=0.6, count=5):
    rootdir = './output/ensemble/'
    list = os.listdir(rootdir)
    path_list = sorted(list, reverse=True)
    import re
    pattern = re.compile(r'.*440_rf')
    path_list = [os.path.join(rootdir, item) for item in path_list if pattern.match(item)]

    val_list = []
    sub_list = {}
    i = 0
    for file in path_list:
        i += 1
        vali = pd.read_hdf(file, 'val')
        vali = vali.groupby('out_id').final_loss.mean()
        vali.name = file
        vali.to_frame()
        # print(vali.shape)
        val_list.append(vali)

    val_all = pd.concat(val_list, axis=1)

    # val_all['file_min'] = val_all.apply(lambda val: val.idxmin(axis=1) , axis=1)
    # file_max = val_all.idxmax(axis=1)
    # val_all['file_max'] = file_max
    val_all['file_min'] = val_all.idxmin(axis=1)
    val_all['min_'] = val_all.min(axis=1)
    val_all['max_'] = val_all.max(axis=1)
    val_all.index.name = 'out_id'
    val_all = val_all.reset_index()

    test = get_score_outid()

    val_all = val_all.merge(test, how='left')

    val_all.sort_values(['min_', 'count'], ascending=False, inplace=True)

    return val_all[(val_all.min_ >= score) & (val_all['count'] >= count)]