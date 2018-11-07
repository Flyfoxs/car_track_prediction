
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from code_felix.utils_.other import replace_invalid_filename_char
from code_felix.utils_.util_log import *
import matplotlib as plt
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
except Exception as e:
    logger.warning(e)


@timed(logger)
def convert_label_encode(sample, exclude=[]):
    try:
        #Label encode
        obj_col = sample.select_dtypes(include=['object']).columns
        obj_col = [ item for item in obj_col if item != 'device' and item not in exclude]
        print(f'{obj_col} will convert to label encode, and fillna with Other')


        sample = sample.apply(lambda x: x.fillna('Other')
                                if x.name in obj_col else x,
                                                      reduce=False)

        label_encode = defaultdict(LabelEncoder)
        sample = sample.apply(lambda x: label_encode[x.name].fit_transform(x.astype(str))
                        if x.name in obj_col else x,
                        reduce=False)


        return sample
    except Exception as e:
        print(f'The columns typs is {sample.dtypes.sort_values()}')
        raise e


import numpy as np
def check_exception(df, index=None):
    df = df.copy(deep=True)
    if index is not None and index in df:
        df.set_index(index,inplace=True)
    df = df.select_dtypes( #include=['float64', 'int'],
                           exclude=['object', 'datetime64[ns]'],)
    try:
        x, y = np.where(np.isinf(df.values) | np.isnan(df.values))
    except Exception as error:
        logger.debug(df.dtypes.sort_values())
        raise error
    if len(x)>0:
        #print(x.min(), x.max()+1, y.min(), y.max()+1)
        df = df.iloc[x.min():(x.max()+3), y.min():(y.max()+3)]
        error_part = df.iloc[:3, :4]
        logger.debug(f'check_exception:\n{error_part}')
        return error_part
    else:
        return pd.DataFrame()


def flat_columns(df, prefix=None):
    df.columns = ['_'.join(drop_empty_item(item)) for item in df.columns]
    if prefix is not None:
        df.columns = [f'{prefix}_{item}' for item in df.columns]
    return df

def drop_empty_item(item):
    return [str(val) for val in item if val is not None and len(str(val))>0]

def save_df(df, path, format='h5'):
    import os
    path = replace_invalid_filename_char(path)
    logger.debug("Save ensmeble result to file:%s" % path)
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    if format=='h5':
        df.to_hdf(path, 'key', index=True)
    return path

