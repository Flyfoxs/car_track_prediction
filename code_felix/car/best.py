from code_felix.car.train_gbdt import *
if __name__ == '__main__':
    # for threshold in [400, 500, 600]:  # 1000,2000 ,300, 400, 500,
    #         for adjust_test in [False, True]:
    #             for max_depth in [4]:
    #                 for model_type in ['lgb']:
    #                 #for estimator in range(50, 300, 50):
    #                     for num_round in [50]:
    #                         for sub in [True, False,]:
    #                             gen_sub(sub, threshold, adjust_test,model_type, num_round = num_round, max_depth=max_depth )
    #

    #gen_sub(False, 400, False, 'xgb', num_round=20, max_depth=6)



    for threshold in [400]:  # 1000,2000 ,300, 400, 500,
        for adjust_test in [False, True]:
            for max_depth in [4]:
                for model_type in ['lgb']:
                    # for estimator in range(50, 300, 50):
                    for num_round in [50,200]:
                        for sub in [False]:
                            gen_sub(sub, threshold, adjust_test, model_type, num_round=num_round, max_depth=max_depth)
                            # exit(0)
    #
    # for threshold in [400, 500, 600]:  # 1000,2000 ,300, 400, 500,
    #         for adjust_test in [False, True]:
    #             for max_depth in [8, 6, 4]:
    #                 for model_type in ['xgb']:
    #                 #for estimator in range(50, 300, 50):
    #                     for num_round in [20,15,10]:
    #                         for sub in [False]:
    #                             gen_sub(sub, threshold, adjust_test,model_type, num_round = num_round, max_depth=max_depth )
    #                             #exit(0)
    #
    #
    #
    # for max_depth in [4]:
    #     for sub in [False, ]:
    #         for adjust_test in [False, True]:
    #             for estimator in [ 10, 50, 100, 200]:
    #                 for threshold in[400,500,600]: #1000,2000 ,300, 400, 500,
    #                     for model_type in ['rf']:
    #                         gen_sub(sub, threshold, adjust_test,model_type, max_depth = max_depth, num_round=estimator,)
    #                         exit(0)
    #
