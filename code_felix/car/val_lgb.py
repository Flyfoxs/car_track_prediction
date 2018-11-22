# 0.42434 + holiday =>0.42796
# 0.42434 + dis_center_0 => 0.42224


from code_felix.car.train_gbdt import *

if __name__ == '__main__':

    for max_depth in [3,4]:
        for num_round in [100,200,300]:
            for split_num in [5, 9]:
                for min_data_in_leaf in [80, 60, 20]:
                    for file in [
                        'new=0gp=638',
                        # 'new=1gp=604',
                        # 'new=2gp=558',
                        # 'new=3gp=627',
                        # 'new=4gp=521',
                        # 'new=5gp=593',
                        # 'new=6gp=580',
                        # 'new=7gp=558',
                        # 'new=8gp=556',
                        'new=9gp=582',
                    ]:
                        gen_sub(file, 500, 0, 'lgb',
                                max_depth=max_depth,
                                num_round=num_round,
                                split_num=split_num,
                                min_data_in_leaf=min_data_in_leaf
                                )


    #
    # for max_depth in [3,4]:
    #     for num_round in [24, 22]:
    #         for file in [100,'all_2' ]:
    #             gen_sub(file, 500, 0, 'lgb', max_depth=max_depth, num_round=num_round)
    #

