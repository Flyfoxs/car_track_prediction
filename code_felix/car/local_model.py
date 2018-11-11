
from code_felix.car.train_gbdt import process_df
from code_felix.car.utils import *

from code_felix.car.utils import get_train_with_adjust_position, get_test_with_adjust_position


for threshold in [ 50, 30, 40,60,70,80,90,  500]:
    for gp in [ 0]:
        for model in ['rf']:
            for deep in [4]:
                        file='all_2'
                        #threshold=500
                        cur_train = f'{DATA_DIR}/train_{file}.csv'
                        cur_test = f'{DATA_DIR}/test_{file}.csv'

                        train = get_train_with_adjust_position(threshold, cur_train)
                        train = analysis_start_zone_id(threshold, cur_train, train)

                        test = get_test_with_adjust_position(threshold, cur_train, cur_test)
                        test = analysis_start_zone_id(threshold, cur_train, test)

                        val_df = process_df(train, test, threshold, gp, model, max_depth=deep, num_round=100)
                        loss = cal_loss_for_df(val_df)
                        if loss:
                            logger.debug(f"=====Loss is {'{:,.5f}'.format(loss)} on {len(test.out_id.drop_duplicates())} cars, "
                                         f"{len(get_feature_columns(gp))} feature({gp}), "
                                         f"thresold:{threshold}, deep:{deep}, model:{model} "                           
                                         f"{len(val_df)} samples")
