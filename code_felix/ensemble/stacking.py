# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from code_felix.car.utils import test_dict
from code_felix.ensemble.checkpoint import ReviewCheckpoint
from code_felix.utils_.util_log import *

from sklearn import datasets
import pandas as pd

from functools import lru_cache
from code_felix.utils_.util_cache_file import *
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras import backend as K
from code_felix.car.utils import *


def get_partition_file_list(partition_name, split_num, top=1, path='./output/ensemble/', ):
    import os
    import re
    file_list = os.listdir(path)
    re_express = f".*split_num{split_num}.*{partition_name}.*h5"
    logger.debug(re_express)
    pattern = re.compile(re_express)
    file_list = [file for file in file_list if pattern.match(file)]

    logger.debug(f"There are {len(file_list)} file in partition:{partition_name}:\n {file_list}")
    file_list = [f'{path}{file}' for file in file_list]
    file_list = sorted(file_list)
    logger.debug(f'Only return {top} of the file list:{file_list[:top]}')
    return file_list[:top]


def baseline_model(input_dim, output_dim):
    # create model
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal'))
    # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model


@lru_cache()
@timed()
def get_train(file_list):
    file_list = file_list.split(',')
    return merge_file(file_list, 'ensemble_train')


@lru_cache()
@timed()
def get_test(file_list):
    file_list = file_list.split(',')
    return merge_file(file_list, 'ensemble_test')


def merge_file(file_list, key):
    """
    val = pd.read_hdf(file, 'val')
    sub = pd.read_hdf(file, 'sub')
    ensemble_test = pd.read_hdf(file, 'ensemble_test')
    ensemble_train = pd.read_hdf(file, 'ensemble_train')
    :param file_list:
    :param key:
    :return:
    """
    df_list = []
    for file in file_list:
        df = pd.read_hdf(file, key)
        df_list.append(df)
    df_merge = pd.concat(df_list, axis=1)
    df_merge.columns = [f'{index}#{col}' for index, col in zip(df_merge.columns, range(0, df_merge.shape[1]))]
    #df_merge.fillna(0, inplace=True)
    return df_merge


@lru_cache()
@timed()
def get_raw_test():
    test = pd.read_csv('./input/del/test_new.csv', delimiter=',', dtype=test_dict)
    return test[['out_id', 'r_key']]


@lru_cache()
def get_label(file_list):
    file_list = file_list.split(',')
    df = pd.read_hdf(file_list[0], 'val')
    df.end_zoneid = df.end_zoneid.astype(int)
    return df


@timed()
def get_outid_inf(file_list, out_id=None):
    # file_list = file_list.split(',')
    train = get_train(file_list)
    label = get_label(file_list)
    test = get_test(file_list)

    if out_id is None:
        logger.debug("Initial cache for sub process")
        return None

    # logger.debug(train.index[:10])
    # logger.debug(label.index[:10])
    # logger.debug(test.index[:10])

    mini_label = label[label.out_id == out_id]
    mini_label.sort_index(inplace=True)

    mini_train = train.loc[mini_label.index]#.copy()
    mini_train = mini_train.dropna(axis=1, how='all').fillna(0)

    raw_test = get_raw_test()
    mini_test = test.loc[raw_test[raw_test.out_id == out_id].r_key]#.copy()
    mini_test = mini_test.dropna(axis=1, how='all').fillna(0)

    from keras.utils import np_utils
    mini_label = pd.DataFrame(np_utils.to_categorical(mini_label.end_zoneid.astype('category').cat.codes),
                              columns=mini_label.end_zoneid.astype('category').cat.categories.astype(int),
                              index=mini_label.index)

    return mini_train,  mini_label, mini_test,

@timed()
def process_partition(partition_list, split_num, top_file):
    for partition in partition_list:
        process_single_partition(partition, split_num, top_file)

@timed()
def process_single_partition(partition, split_num, top_file):
    file_list = get_partition_file_list(partition, split_num, top_file)
    file_list = ','.join(file_list)
    # print(get_label(file_list))
    label = get_label(file_list)
    out_id_list = label.groupby('out_id')['out_id'].count().sort_values(ascending=False)
    #logger.debug(f'Out_id:{out_id_list.index[:5]}')
    logger.debug(type(out_id_list.index))
    sub_list = []
    #val_list = []
    # initial cache
    get_outid_inf(file_list)
    out_id_total = len(out_id_list.index)
    i = 0
    for out_id in out_id_list.index : # ['868260020955218', '891691612019981'] : #
        i += 1
        # Start a new process to avoid the memory leak in keras
        from multiprocessing import Pool as ThreadPool
        from functools import partial
        learning_ex = partial(learning, file_list=file_list, partition = partition)
        pool = ThreadPool(processes=1)

        results = pool.map(learning_ex, [out_id])
        pool.close()
        pool.join()
        logger.debug(f'{type(results)}:{len(results)}')
        sub_list.extend(results)
        #val_list.extend([res[1] for res in results])
        logger.debug(f'Process status for this partition#{partition}::{i}/{out_id_total}')
        # logger.debug(f"sub_list:{len(sub_list)}, val_list:{len(val_list)}")
    # logger.debug(sub.idxmax(axis=1).head(2))
    #val_partition = pd.concat(val_list).sort_index()
    label = get_label(file_list).sort_index()
    # logger.debug(f'{type(val_partition.idxmax(axis=1))}, {type(label.end_zoneid)}')
    # logger.debug(f'{val_partition.idxmax(axis=1).shape}, {label.end_zoneid.loc[val_partition.index].shape}')

    #logger.debug(val_partition.idxmax(axis=1)[:20])

    #logger.debug(label.end_zoneid[:20])

    # logger.debug(f'{val_partition.idxmax(axis=1).dtype}, {label.end_zoneid.dtype}')


    # accuracy_partition = np.mean(val_partition.idxmax(axis=1).values.astype(int) == label.end_zoneid.loc[val_partition.index].values.astype(int))
    # accuracy_partition = round(accuracy_partition, 4)

    sub_partition = pd.concat(sub_list).sort_index()
    logger.debug( f'The accuracy for partition:{partition}, split_num:{split_num}, records:{len(sub_partition)}')

    path = f'./output/ensemble/level2/st_{partition}_{out_id_total}_{len(sub_partition)}_{split_num}.h5'

    save_result_partition(None, sub_partition, path)
    return path


@timed()
def learning( out_id, file_list, partition  ):
    train, label, test = get_outid_inf(file_list, out_id)
    logger.debug(f'Get data for out_id:{out_id}, train:{train.shape}, label:{label.shape}, test:{test.shape}')

    sub_list=[]
    if label.shape[1] == 1:
        val = pd.DataFrame(1, columns=label.columns, index=train.index)
        sub =  pd.DataFrame(1, columns=label.columns, index=test.index)
        logger.debug(f'The final accuracy for out_id({len(train)}):{out_id} is 1, only 1 label')

        return sub #, val
    else:
        start = time.time()
        model = RandomForestClassifier(max_depth=4,
                                 n_estimators = 100,
                                 n_jobs=4,
                                 random_state=0)

        label_new = label.idxmax(axis=1)
        label_new = pd.Categorical(label_new).codes
        model.fit(train, label_new)

        sub_folder = pd.DataFrame(model.predict_proba(test), columns=label.columns, index=test.index)
        sub_list.append(sub_folder)
        duration = time.time() - start
        logging.info('cost:%7.2f sec: ===%s' % (duration, f'Fit the out_id({len(train)}):{out_id}, {partition}'))

        sub_concat = pd.concat(sub_list)
        sub_concat.index.name='r_key'

        logger.debug(f'======{type(sub_concat)}')
        return sub_concat


def wider_model(**kw):
    model = LinearRegression()
        # RandomForestClassifier(max_depth=kw['max_depth'],
        #                          n_estimators = kw['num_round'],
        #                          n_jobs=4,
        #                          random_state=0)
    return model

if __name__ == '__main__':
    # file_list = get_partition_file_list('9gp', 5)
    #label = get_label(file_list)
    #logger.debug(label.out_id.head(3))
    # train, test, label = get_outid_inf(file_list, '4A23256745CBA3B0')
    # print(train.shape, test.shape, label.shape)
    import sys
    if len(sys.argv)>=2:
        partition_list  = sys.argv[1:] #['9gp']
        partition_list = [f'{i}gp' for i in partition_list]
        logger.debug(f'Partition_list:{partition_list}')
    else:
        logger.debug("Need input parittion list")



   # paras = process_partition(['9gp'], 5, top_file=1) #3529
    paras = process_partition(partition_list, 5, top_file=2) #3765

