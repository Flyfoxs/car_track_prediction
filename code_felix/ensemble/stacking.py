# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense

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
    df_merge.columns = range(0, df_merge.shape[1])
    df_merge.fillna(0, inplace=True)
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

    raw_test = get_raw_test()
    mini_test = test.loc[raw_test[raw_test.out_id == out_id].r_key]#.copy()

    from keras.utils import np_utils
    mini_label = pd.DataFrame(np_utils.to_categorical(mini_label.end_zoneid.astype('category').cat.codes),
                              columns=mini_label.end_zoneid.astype('category').cat.categories.astype(int),
                              index=mini_label.index)

    return mini_train,  mini_label, mini_test,

@timed()

def process_partition(partition_list, split_num, top_file):
    for partition in partition_list:
        return process_single_partition(partition, split_num, top_file)

def process_single_partition(partition, split_num, top_file):
    file_list = get_partition_file_list(partition, split_num, top_file)
    file_list = ','.join(file_list)
    # print(get_label(file_list))
    label = get_label(file_list)
    out_id_list = label.groupby('out_id')['out_id'].count().sort_values(ascending=False)
    #logger.debug(f'Out_id:{out_id_list.index[:5]}')
    logger.debug(type(out_id_list.index))
    sub_list = []
    val_list = []
    # initial cache
    get_outid_inf(file_list)
    out_id_total = len(out_id_list.index)
    i = 0
    for out_id in out_id_list.index :
        i += 1
        # Start a new process to avoid the memory leak in keras
        from multiprocessing import Pool as ThreadPool
        from functools import partial
        learning_ex = partial(learning, file_list=file_list, split_num=split_num)
        pool = ThreadPool(processes=1)

        results = pool.map(learning_ex, [out_id])
        pool.close()
        pool.join()
        sub_list.extend([res[0] for res in results])
        val_list.extend([res[1] for res in results])
        logger.debug(f'Process status for this partition#{partition}::{i}/{out_id_total}')
        # logger.debug(f"sub_list:{len(sub_list)}, val_list:{len(val_list)}")
    # logger.debug(sub.idxmax(axis=1).head(2))
    val_partition = pd.concat(val_list).sort_index()
    label = get_label(file_list).sort_index()
    logger.debug(f'{type(val_partition.idxmax(axis=1))}, {type(label.end_zoneid)}')
    logger.debug(f'{val_partition.idxmax(axis=1).shape}, {label.end_zoneid.loc[val_partition.index].shape}')

    logger.debug(val_partition.idxmax(axis=1)[:20])

    logger.debug(label.end_zoneid[:20])

    logger.debug(f'{val_partition.idxmax(axis=1).dtype}, {label.end_zoneid.dtype}')


    accuracy_partition = np.mean(val_partition.idxmax(axis=1).values.astype(int) == label.end_zoneid.loc[val_partition.index].values.astype(int))
    accuracy_partition = round(accuracy_partition, 4)
    logger.debug(f'The accuracy for partition:{partition}, split_num:{split_num}, records:{len(val_partition)} is {accuracy_partition}')

    path = f'./output/sub/{partition}_{split_num}_{accuracy_partition}.h5'

    sub_partition = pd.concat(sub_list).sort_index()
    save_result_partition(val_partition, sub_partition, path)
    return path


@timed()
def learning( out_id, file_list, split_num,   ):
    train, label, test = get_outid_inf(file_list, out_id)
    logger.debug(f'Get data for out_id:{out_id}, train:{train.shape}, label:{label.shape}, test:{test.shape}')
    kf = KFold(n_splits=split_num, shuffle=True, random_state=777)

    accuracy_list = []
    sub_list = []
    for folder, (train_index, val_index) in enumerate(kf.split(train)):

        model = wider_model(train.shape[1], label.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=50, )

        model_file = './output/model/checkpoint.h5'
        # check_best = ModelCheckpoint(filepath=model_file,
        #                              monitor='val_loss', verbose=1,
        #                              save_best_only=True, mode='min')

        #logger.debug(f'Train:{train.shape}, Label:{label.shape}, folder:{folder}, {val_index}')

        check_best = ReviewCheckpoint(model_file, folder, train.iloc[val_index], label.iloc[val_index])
        start = time.time()
        history = model.fit(train.iloc[train_index], label.iloc[train_index],
                            validation_data=(train.iloc[val_index], label.iloc[val_index]),
                            callbacks=[check_best,
                                       early_stop,
                                       # reduce,
                                       ],
                            batch_size=32,
                            epochs=100,
                            )
        duration = time.time() - start
        logging.info('cost:%7.2f sec: ===%s' % (duration, f'Fit the out_id({len(train_index)}):{out_id}, folder:{folder}'))

        from keras import models
        best_model = models.load_model(model_file)
        val = train.iloc[val_index]
        acc = pd.DataFrame(best_model.predict(val), columns=label.columns, index=val.index)
        accuracy_list.append(acc)
        #accuracy_rate = np.mean(np.argmax(acc.values, axis=1) == np.argmax(label.values[val_index], axis=1))
        accuracy_rate = np.mean(acc.idxmax(axis=1) == label.iloc[val_index].idxmax(axis=1))
        logger.debug(f'Folder:{folder}, The best accuracy for out_id:{out_id} is {round(accuracy_rate,4)}')

        best_epoch = np.array(history.history['val_loss']).argmin() + 1
        best_score = np.array(history.history['val_loss']).min()

        sub_folder = pd.DataFrame(model.predict(test), columns=label.columns, index=test.index)
        sub_list.append(sub_folder)

        #K.clear_session()

    val_all = pd.concat(accuracy_list).sort_index()
    label = label.sort_index()
    accuracy = np.mean(np.argmax(val_all.values, axis=1) == np.argmax(label.values, axis=1))
    logger.debug(f'The final accuracy for out_id({len(train)}):{out_id} is <<{round(accuracy,4)}>>')

    sub_merge = pd.concat(sub_list)
    sub_merge.index.name='r_key'
    sub_mean = sub_merge.reset_index().groupby('r_key').mean()

    return sub_mean, val_all


def wider_model(input_dim, output_dim):
    # create model
    model = Sequential()
    model.add(Dense(int(input_dim*4), input_dim=input_dim, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(int(output_dim*4), kernel_initializer='normal', activation='tanh'), )
    model.add(Dense(int(output_dim*4), kernel_initializer='normal', activation='tanh'), )
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.summary()
    return model

if __name__ == '__main__':
    # file_list = get_partition_file_list('9gp', 5)
    #label = get_label(file_list)
    #logger.debug(label.out_id.head(3))
    # train, test, label = get_outid_inf(file_list, '4A23256745CBA3B0')
    # print(train.shape, test.shape, label.shape)


   # paras = process_partition(['9gp'], 5, top_file=1) #3529
    paras = process_partition(['9gp'], 5, top_file=2) #3765

