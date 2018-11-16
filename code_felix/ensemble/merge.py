from code_felix.car.utils import *
import os.path

def get_file_name(file_list, comments=''):
    file_list = [ item.split('_')[0].replace('0.','') for item in file_list]
    file_list = sorted(file_list)
    file = '_'.join(file_list)
    return f'./output/merge_{comments}_{file}.csv'

if __name__ == '__main__':
    rootdir = './output/ensemble/'
    list = os.listdir(rootdir)
    # path_list = sorted(list, reverse=True)
    # import re
    # pattern = re.compile(r'.*43959.*h5$')

    path_list =[
        # '0.43957_rf_70_kwmax_depth4num_round100model_typerfgp0threshold70fileall_2.h5',
        # '0.43959_rf_40_kwmax_depth4num_round100model_typerfgp0threshold40fileall_2.h5',
        # '0.43960_rf_500_kwmax_depth4num_round100model_typerfgp0threshold500fileall_2.h5',
       #Bad '0.790_rf_2000_kwmax_depth4num_round100model_typerfgp0threshold2000fileworse.h5',

        ########## 20% val ###########
        # '0.41938_rf_550_kwmax_depth4num_round100model_typerfgp0threshold550fileall_3.h5',
        # #'0.41957_rf_500_kwmax_depth4num_round100model_typerfgp0threshold500fileall_3.h5',
        #

        '0.29215_rf_500_kwmax_depth4num_round100model_typerfgp0threshold500file0gp=1242.h5',
        '0.39439_rf_500_kwmax_depth4num_round100model_typerfgp0threshold500file1gp=1185.h5',
        '0.43971_rf_500_kwmax_depth4num_round100model_typerfgp0threshold500file2gp=1114.h5',
        '0.47908_rf_500_kwmax_depth4num_round100model_typerfgp0threshold500file3gp=1138.h5',
        '0.55279_rf_500_kwmax_depth4num_round100model_typerfgp0threshold500file4gp=1138.h5',


    ]

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
    test = get_time_extend('./input/test_new.csv')
    sub_df = pd.DataFrame(index=test.r_key).join(sub_df)

    file_name = get_file_name(file_list, new_loss)
    logger.debug(file_name)
    sub_df.to_csv(file_name)

