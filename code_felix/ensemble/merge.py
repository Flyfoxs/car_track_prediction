from code_felix.car.utils import *
import os.path

def get_file_name(comments):

    return f'./output/merge_{comments}.csv'

if __name__ == '__main__':
    rootdir = './output/ensemble/1level/'
    list = os.listdir(rootdir)
    # path_list = sorted(list, reverse=True)
    # import re
    # pattern = re.compile(r'.*43959.*h5$')

    # path_list =[
    #
    #     '0.20596_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=0gp=638.h5',
    #     '0.29345_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=1gp=604.h5',
    #     '0.34455_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=2gp=558.h5',
    #     '0.37130_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=3gp=627.h5',
    #     '0.38894_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=4gp=521.h5',
    #     '0.40405_rf_500_kwmax_depth4num_round100split_num9model_typerfgp0threshold500filenew=5gp=593.h5',
    #     '0.40483_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=5gp=593.h5',
    #     '0.42350_rf_500_kwmax_depth4num_round100split_num9model_typerfgp0threshold500filenew=6gp=580.h5',
    #     '0.42419_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=6gp=580.h5',
    #     '0.46018_rf_500_kwmax_depth4num_round100split_num9model_typerfgp0threshold500filenew=7gp=558.h5',
    #     '0.46175_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=7gp=558.h5',
    #     '0.49107_rf_500_kwmax_depth4num_round100split_num9model_typerfgp0threshold500filenew=8gp=556.h5',
    #     '0.49267_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=8gp=556.h5',
    #     '0.55944_rf_500_kwmax_depth4num_round100split_num9model_typerfgp0threshold500filenew=9gp=582.h5',
    #     '0.56076_rf_500_kwmax_depth4num_round100split_num5model_typerfgp0threshold500filenew=9gp=582.h5',
    #
    # ]

    path_list = [item for item in list if item.endswith("h5")]

    path_list = [os.path.join(rootdir, item) for item in path_list]

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
    #file_max = val_all.idxmax(axis=1)
    #val_all['file_max'] = file_max
    val_all['file_min'] = val_all.idxmin(axis=1)
    val_all['min_'] = val_all.min(axis=1)
    #val_all['max_'] = val_all.max(axis=1)
    val_all.index.name = 'out_id'
    val_all = val_all.reset_index()

    new_loss = round(val_all.min_.mean(),6)

    sub_list = []
    file_list = []
    gp = val_all.groupby('file_min')
    for file, item in gp:
        logger.debug(f'{str(len(item)).rjust(4,"0")}:{file}')
        temp_sub = pd.read_hdf(file, 'sub')
        sub = temp_sub[temp_sub.out_id.isin(item.out_id)]
        file_name = os.path.basename(file)
        file_list.append(file_name)
        sub_list.append(sub)

    sub_df = pd.concat(sub_list)[['predict_lat', 'predict_lon']]
    sub_df.columns = ['end_lat', 'end_lon']
    sub_df.index.name = 'r_key'
    test = get_time_geo_extend('./input/test_new.csv')
    sub_df = pd.DataFrame(index=test.r_key).join(sub_df)

    file_name = get_file_name(f'{new_loss}_merge_1level_{len(file_list)}', )
    logger.debug(file_name)
    sub_df.to_csv(file_name)

