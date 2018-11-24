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


# if __name__ == '__main__':
rootdir = './output/ensemble/level1/'
list = os.listdir(rootdir)
# path_list = sorted(list, reverse=True)
# import re
# pattern = re.compile(r'.*43959.*h5$')

path_list = [

    './output/ensemble/level2/st_0gp_536_6302_5.h5',
    './output/ensemble/level2/st_1gp_527_5912_5.h5',
    './output/ensemble/level2/st_2gp_491_5654_5.h5',
    './output/ensemble/level2/st_3gp_539_6004_5.h5',
    './output/ensemble/level2/st_4gp_442_5071_5.h5',
    './output/ensemble/level2/st_5gp_523_6143_5.h5',
    './output/ensemble/level2/st_6gp_488_5531_5.h5',
    './output/ensemble/level2/st_7gp_490_5661_5.h5',
    './output/ensemble/level2/st_8gp_492_5563_5.h5',
    './output/ensemble/level2/st_9gp_505_6256_5.h5',

]

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
path = './output/sub/st_sub.csv'
sub_df.to_csv(path)

logger.debug(f'Save sub to file:{path}')


