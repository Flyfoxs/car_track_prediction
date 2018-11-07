# 0.42434 + holiday =>0.42796
# 0.42434 + dis_center_0 => 0.42224
from code_felix.car.train_rf import *

from code_felix.car.train_gbdt import *

if __name__ == '__main__':



    for num_round in [8, 10]:
        for file in ['all',True]:
            gen_sub(file, 500, 0, 'xgb', max_depth=4, num_round=num_round)

    # for num_round in [8, 10]:
    #     gen_sub(True, 500, 0, 'xgb', max_depth=4, num_round=num_round)
