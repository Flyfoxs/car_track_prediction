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
#'./output/ensemble/level1/0.00000_rf_gp0_400_kwmax_depth4num_round100split_num1precision4model_typerfgp0threshold400filenew=3geo=1026.h5',
#'./output/ensemble/level1/0.00000_rf_gp0_400_kwmax_depth4num_round100split_num1precision5model_typerfgp0threshold400filenew=5geo=148.h5',
'./output/ensemble/level1/0.00000_rf_gp0_200_kwmax_depth4num_round100split_num1model_typerfgp0threshold200filenew=0geo=423.h5',
'./output/ensemble/level1/0.00000_rf_gp0_400_kwmax_depth4num_round100split_num1model_typerfgp0threshold400filenew=0geo=423.h5',

'./output/ensemble/level1/0.00000_rf_gp0_400_kwmax_depth4num_round100split_num1model_typerfgp0threshold400filenew=1geo=1801.h5',
'./output/ensemble/level1/0.00000_rf_gp0_400_kwmax_depth4num_round100split_num1model_typerfgp0threshold400filenew=2geo=2101.h5',
'./output/ensemble/level1/0.00000_rf_gp0_400_kwmax_depth4num_round100split_num1model_typerfgp0threshold400filenew=3geo=1026.h5',
'./output/ensemble/level1/0.00000_rf_gp0_400_kwmax_depth4num_round100split_num1model_typerfgp0threshold400filenew=4geo=318.h5',
'./output/ensemble/level1/0.00000_rf_gp0_400_kwmax_depth4num_round100split_num1model_typerfgp0threshold400filenew=5geo=148.h5',

]

fina_sub = pd.DataFrame()

for file in path_list:
    sub = read_sub(file)
    sub.set_index('r_key', inplace=True)
    fina_sub = fina_sub.combine_first(sub)

print(fina_sub.shape)

test = get_time_geo_extend(test_file)
sub_df = pd.DataFrame(index=test.r_key).join(fina_sub)

sub_df.columns = ['end_lat','end_lon']
path = './output/sub/concat_sub_.csv'
sub_df.to_csv(path)

logger.debug(f'Save sub to file:{path}')


