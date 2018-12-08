# 0.42434
from code_felix.car.train_gbdt import *


    #0.42224+500threshold = 0.41796
    # gen_sub(100, 500, 0,'rf', max_depth=4, num_round=100)

    # for threshold in [500]:
    #     for sub in sorted(['0gp2704', '1gp1144', '2gp1637', '3gp237', '4gp95',], reverse=False):
    #         for deep in [4, 5, 6]:
    #             for feature_gp in [0]:
    #                 gen_sub(sub, threshold, 0, 'rf', max_depth=deep, num_round=100)



def optimize_rf(args):
    deep = args['deep']
    num_round = args['num_round']
    threshold = args['threshold']

    sub = 'new=2geo=2101'
    split_num = 5
    return gen_sub(sub, threshold, feature_gp, 'rf', max_depth=deep, num_round=num_round, split_num= split_num)

from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

if __name__ == '__main__':
    space = {"deep": hp.choice("max_depth", range(3,8)),
             "num_round": hp.randint("n_estimators", 65),  # [0,1,2,3,4,5] -> [50,]
             "threshold": hp.choice("threshold", range(300, 500, 50))
             #"threshold": hp.randint("threshold", 400),
             }

trials = Trials()
best = fmin(optimize_rf, space, algo=tpe.suggest, max_evals=10, trials=trials)


for score,paras in dict( trials.losses() , [item.get('misc').get('vals') for item in trials.trials]):
    logger.debug(f'score:{score}:para:{paras}')