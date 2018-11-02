from code_felix.car.train_gbdt import *
if __name__ == '__main__':
    for threshold in [400, 500, 600]:  # 1000,2000 ,300, 400, 500,
            for adjust_test in [False, True]:
                for max_depth in [4]:
                    for model_type in ['lgb']:
                    #for estimator in range(50, 300, 50):
                        for num_round in [50]:
                            for sub in [True, False,]:
                                gen_sub(sub, threshold, adjust_test,model_type, num_round = num_round, max_depth=max_depth )
                                ##exit(0)


