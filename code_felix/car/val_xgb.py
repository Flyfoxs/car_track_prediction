# 0.42434 + holiday =>0.42796
# 0.42434 + dis_center_0 => 0.42224
from code_felix.car.train_rf import *

from code_felix.car.train_gbdt import *

if __name__ == '__main__':
    for deep in [4, 5, 6]:
        for threshold in [500]:
            for split_num in [5, 9]:
                for sub in sorted(['0gp2704', '1gp1144', '2gp1637', '3gp237', '4gp95',], reverse=True):
                    for feature_gp in [0]:
                        gen_sub(sub, threshold, 0, 'xgb', max_depth=deep, num_round=100, split_num=split_num)

    # for num_round in [10, 100]:
    #     for file in ['worse']:
    #         gen_sub(file, 2000, 0, 'xgb', max_depth=4, num_round=num_round)

    #
    # for num_round in [8, 10]:
    #     for file in [100,  'all_2']:
    #         gen_sub(file, 500, 0, 'xgb', max_depth=4, num_round=num_round)

    # for num_round in [8, 10]:
    #     gen_sub(True, 500, 0, 'xgb', max_depth=4, num_round=num_round)
