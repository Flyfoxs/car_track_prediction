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

    import sys
    if len(sys.argv) > 1 :
        feature_gp_list = sys.argv[1:]
    else:
        feature_gp_list = [0]


    for deep in [4]:
        for num_round in [50,60,70]:
                    for feature_gp in feature_gp_list:
                        for split_num in [5]:
                            for sub , threshold in sorted([
                                    ('new=0geo=423',  200),
                                    ('new=1geo=1801', 300),
                                    ('new=2geo=2101', 400),
                                    ('new=3geo=1026', 450),
                                    ('new=4geo=318', 550),
                                    ('new=5geo=148', 450),
                            ], reverse=True):
                                 gen_sub(sub, threshold, feature_gp, 'rf', max_depth=deep, num_round=num_round, split_num= split_num)
