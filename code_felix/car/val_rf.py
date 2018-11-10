# 0.42434
from code_felix.car.train_gbdt import *

if __name__ == '__main__':
    #0.42224+500threshold = 0.41796
    # gen_sub(100, 500, 0,'rf', max_depth=4, num_round=100)

    for threshold in range(1000, 7000, 1000):
        for sub in ['worse']:
            for feature_gp in [0, 1,2,3]:
                gen_sub(sub, threshold, 0, 'rf', max_depth=4, num_round=100)

    # gen_sub('all', 500, 2, max_depth=4)
    #gen_sub(True, 600, 2, max_depth=4)

    # for threshold in [500]:
    #     for sub in [100, 'all']:
    #         for feature_gp in range(0, 4):
    #             gen_sub(sub, threshold, feature_gp, max_depth=4)
