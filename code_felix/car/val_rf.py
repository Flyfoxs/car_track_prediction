# 0.42434
from code_felix.car.train_gbdt import *

if __name__ == '__main__':
    #0.42224+500threshold = 0.41796
    # gen_sub(100, 500, 0,'rf', max_depth=4, num_round=100)

    # for threshold in [500]:
    #     for sub in sorted(['0gp2704', '1gp1144', '2gp1637', '3gp237', '4gp95',], reverse=False):
    #         for deep in [4, 5, 6]:
    #             for feature_gp in [0]:
    #                 gen_sub(sub, threshold, 0, 'rf', max_depth=deep, num_round=100)


    for threshold in [500]:
        for sub in sorted(['0gp=1242','1gp=1185','2gp=1114','3gp=1138','4gp=1138',], reverse=False):
            for deep in [4]:
                for feature_gp in [0]:
                    gen_sub(sub, threshold, 0, 'rf', max_depth=deep, num_round=100)

    # for threshold in sorted([  40,70,500, 30, 50, 450, 550, 2000], reverse=True):
    #     for sub in ['all_3']:
    #         for feature_gp in [0]:
    #             gen_sub(sub, threshold, 0, 'rf', max_depth=4, num_round=100)

    # for threshold in [ 20, 30, 40, 50,60,70,80, 90, 500]:
    #     for sub in ['all_2']:
    #         for feature_gp in [0]:
    #             gen_sub(sub, threshold, 0, 'rf', max_depth=4, num_round=100)

    # gen_sub('all', 500, 2, max_depth=4)
    #gen_sub(True, 600, 2, max_depth=4)

    # for threshold in [500]:
    #     for sub in [100, 'all']:
    #         for feature_gp in range(0, 4):
    #             gen_sub(sub, threshold, feature_gp, max_depth=4)
