
from code_felix.car.train_gbdt import process_df
from code_felix.car.utils import *

from code_felix.car.utils import get_train_with_adjust_position, get_test_with_adjust_position

#rm ./cache/*3gp237* ; rm ./cache/*4gp95*; nohup python code_felix/car/local_model.py >> local_2.log 2>&1 &
for threshold in [500]:
    for gp in [ 0]:
        for model in [  'lgb', ]:
            for deep in [4]:
                   for  file in [ '4gp95']: #,,,
                       for n_neighbors in [19]:
                           for p in [5]:
                               for split_num in [5]:
                                   for min_data_in_leaf in[80, 60, 20] :
                                    #threshold=500
                                    cur_train = f'{DATA_DIR}/train_{file}.csv'
                                    cur_test = f'{DATA_DIR}/test_{file}.csv'

                                    train = get_train_with_adjust_position(threshold, cur_train)
                                    #train = analysis_start_zone_id(threshold, cur_train, train)

                                    test = get_test_with_adjust_position(threshold, cur_train, cur_test)
                                    #test = analysis_start_zone_id(threshold, cur_train, test)

                                    #n_neighbors = kw['n_neighbors']  # 15
                                    #weights = kw['weights']  # 'distance'
                                    predict_df, val_df, _, _ = process_df(train, test, threshold, gp, model,
                                                        max_depth=deep, num_round=100,
                                                        n_neighbors = n_neighbors,
                                                        split_num= split_num,
                                                        min_data_in_leaf=min_data_in_leaf
                                                                    )
                                    loss , accuracy = cal_loss_for_df(val_df)

                                    feature_gp = 'knn' if model =='knn' else gp

                                    if loss:
                                        logger.debug(f"=====Loss:{'{:,.5f}'.format(loss)}, accuracy:{accuracy} on {len(train.out_id.drop_duplicates())} cars, "
                                                     f"{len(get_feature_columns(feature_gp))} feature({feature_gp} file:{file}), "
                                                     f"thresold:{threshold}, deep:{deep}, model:{model}, "
                                                     f"nb:{n_neighbors},  min_data_in_leaf:{min_data_in_leaf}, split_num:{split_num} "                           
                                                     f"{len(predict_df)} samples")
                                    else:
                                        logger.debug(f'=====Loss is None, split_num:{split_num}')


