
from code_felix.car.train_gbdt import process_df
from code_felix.car.utils import *

from code_felix.car.utils import get_train_with_adjust_position, get_test_with_adjust_position


for threshold in [ 100, 500, 1000, 2000, 3000]:
    for gp in [0, 1, 2]:
        for model in ['rf', 'lgb', 'xgb']:
            file='worse'
            #threshold=500
            cur_train = f'{DATA_DIR}/train_{file}.csv'
            cur_test = f'{DATA_DIR}/test_{file}.csv'
            train = get_train_with_adjust_position(threshold, cur_train)
            test = get_test_with_adjust_position(threshold, cur_train, cur_test)

            val_df = process_df(train, test, threshold, gp, 'rf', max_depth=4, num_round=100)
            loss = cal_loss_for_df(val_df)
            if loss:
                logger.debug(f"=====Loss is {'{:,.5f}'.format(loss)} on {len(test.out_id.drop_duplicates())} cars, "
                             f"{len(get_feature_columns(gp))} feature, "
                             f"{len(val_df)} samples")
