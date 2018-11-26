from code_felix.car.utils import *
from code_felix.ensemble.stacking import *
from code_felix.car.distance_reduce import *
import os.path


def read_sub(file):
    sub = pd.read_hdf(file, 'sub').reset_index()
    sub.drop('index', axis=1, inplace=True, errors='ignore')

    if 'predict_lat' in sub:
        sub.reset_index(inplace=True)
        sub = sub[['r_key', 'predict_lat', 'predict_lon']]
        logger.debug(sub.shape)
        return sub
    else:
        logger.debug(f'=======0\n{sub.head()}')
        return fill_address_info(sub)


def fill_address_info(sub):
    if 'predict_lat' in sub:
        return sub
    else:
        sub.set_index(['out_id', 'r_key', ], inplace=True)

        logger.debug(f'=======0.5{sub.head()}')
        sub = sub.idxmax(axis=1).reset_index()
        sub.columns = ['out_id', 'r_key', 'zoneid']
        logger.debug(f'=======1{sub.head()}')

        address = reduce_address(500, train_file)
        address = address.drop_duplicates(['out_id', 'zoneid'])
        sub = pd.merge(sub, address, on=['out_id', 'zoneid'], how='left')
        sub = sub[['r_key', 'lat', 'lon']]

        logger.debug(f'=======2{sub.head()}')

        sub.columns = ['r_key', 'predict_lat', 'predict_lon']

        return sub



import sys
express = sys.argv[1]

rootdir = './output/ensemble/level1/'
list = os.listdir(rootdir)
# path_list = sorted(list, reverse=True)
import re
pattern = re.compile(express)

path_list = [f'{rootdir}{file}' for file in list if pattern.match(file)]



logger.debug(f'Base on {express}, The input file list{len(path_list)} is:\n {path_list}')

sub_list = []
for file in path_list:
    sub = read_sub(file)
    sub_list.append(sub)
fina_sub = pd.concat(sub_list)
fina_sub.set_index('r_key', inplace=True)
print(fina_sub.shape)

test = get_time_extend(test_file)
sub_df = pd.DataFrame(index=test.r_key).join(fina_sub)

sub_df.columns = ['end_lat','end_lon']
path = f"./output/sub/concat_sub_{express.replace('*','_')}.csv"
sub_df.to_csv(path)

logger.debug(f'Save sub to file:{path}')


