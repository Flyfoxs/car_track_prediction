# 0.42434 + holiday =>0.42796
# 0.42434 + dis_center_0 => 0.42224


from code_felix.car.train_gbdt import *

if __name__ == '__main__':

    for max_depth in [3,4]:
        for num_round in [24]:
            for file in ['worse' ]:
                gen_sub(file, 2000, 0, 'lgb', max_depth=max_depth, num_round=num_round)


    #
    # for max_depth in [3,4]:
    #     for num_round in [24, 22]:
    #         for file in [100,'all_2' ]:
    #             gen_sub(file, 500, 0, 'lgb', max_depth=max_depth, num_round=num_round)
    #

