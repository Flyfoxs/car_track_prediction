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


def split_2_group(gp_num=5):
    train = get_train_with_adjust_position(500, './input/train_new.csv')
    gp_train = train.groupby('out_id')['end_zoneid'].nunique().to_frame().reset_index()
    #gp_train['group'] = pd.cut(gp_train.end_zoneid.astype('int'), [0, 40, 50, 80, 100, 500])
    gp_train['group'] = pd.qcut(gp_train.end_zoneid.astype('int'), gp_num)
    gp_train['group_sn'] = gp_train['group'].cat.codes
    gp_train.head()
    return  gp_train

def split_2_file():
    train = pd.read_csv('./input/del/train_new.csv', delimiter=',', dtype=train_dict)
    gp = split_2_group()
    for gp_name, outid_list  in gp.groupby('group_sn'):
        gp_count = len(outid_list)
        validate_list = []
        i = 0
        for out_id in outid_list.out_id:
            i += 1
            if i % 100 == 0:
                logger.debug(f'{out_id}, {i}, {len(outid_list)}')
            mini = train.loc[train.out_id == out_id]
            count_val = len(mini) // 5
            validate_list.append(mini.head(count_val))

        validate = pd.concat(validate_list)

        train_train_file = f'{DATA_DIR}/train_{gp_name}gp={gp_count}.csv'
        train_validate_file = f'{DATA_DIR}/test_{gp_name}gp={gp_count}.csv'

        validate.to_csv(train_validate_file, index=None)
        # logger.debug(len(validate), len(validate.out_id.drop_duplicates()))

        logger.debug(gp_count)
        train_train = train[(~train.index.isin(validate.index))
                            & (train.out_id.isin(outid_list.out_id))]
        train_train.to_csv(train_train_file, index=None)

def split_2_new():
    train = pd.read_csv('./input/del/train_new.csv', delimiter=',', dtype=train_dict)
    test = pd.read_csv('./input/del/test_new.csv', delimiter=',', dtype=test_dict)
    gp = split_2_group(10)
    for gp_name, outid_list in gp.groupby('group_sn'):
        gp_count = len(outid_list)

        gp_train = train[train.out_id.isin(outid_list.out_id) ]
        gp_test = test[test.out_id.isin(outid_list.out_id) ]

        train_file = f'{DATA_DIR}/train_new={gp_name}gp={gp_count}.csv'
        test_file = f'{DATA_DIR}/test_new={gp_name}gp={gp_count}.csv'

        gp_train.to_csv(train_file, index=None)
        gp_test.to_csv(test_file, index=None)

        # logger.debug(len(validate), len(validate.out_id.drop_duplicates()))




if __name__ == '__main__':
    split_2_new()
    #split_2_file()