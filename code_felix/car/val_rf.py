# 0.42434
from code_felix.car.train_gbdt import *

if __name__ == '__main__':
    #0.42224
    gen_sub(100, 500, 0,'rf', max_depth=4, num_round=100)

    for threshold in [500, 550, 450]:
        for sub in [True,'all']:
            for feature_gp in [3, 0]:
                gen_sub(sub, threshold, feature_gp, 'rf', max_depth=4, num_round=100)


    # gen_sub('all', 500, 2, max_depth=4)
    #gen_sub(True, 600, 2, max_depth=4)

    # for threshold in [500]:
    #     for sub in [100, 'all']:
    #         for feature_gp in range(0, 4):
    #             gen_sub(sub, threshold, feature_gp, max_depth=4)
