# 0.42434
from code_felix.car.train_gbdt import *
import time

    #0.42224+500threshold = 0.41796
    # gen_sub(100, 500, 0,'rf', max_depth=4, num_round=100)

    # for threshold in [500]:
    #     for sub in sorted(['0gp2704', '1gp1144', '2gp1637', '3gp237', '4gp95',], reverse=False):
    #         for deep in [4, 5, 6]:
    #             for feature_gp in [0]:
    #                 gen_sub(sub, threshold, 0, 'rf', max_depth=deep, num_round=100)



def optimize_rf(args):
    input_args = locals()
    deep = args['deep']
    num_round = args['num_round']
    threshold = args['threshold']

    sub = 'new=2geo=2101'
    split_num = 5
    start = time.time()
    loss = gen_sub(sub, threshold, feature_gp, 'rf', max_depth=deep, num_round=num_round, split_num= split_num)
    duration = str(int(time.time() - start))
    logger.debug(f'Get Loss:{"%8.6f"%score} with paras:{input_args}')
    return  {
        'loss': loss,
        'status': STATUS_OK,
        # -- store other results like this
        #'eval_time': time.time(),
        #'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # -- attachments are handled differently
        'attachments': {"message": f'{input_args}, cost:{duration.rjust(5," ")} sec', }
        }
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

if __name__ == '__main__':
    space = {"deep":      hp.choice("max_depth", [3,4,5]),
             "num_round": hp.choice("n_estimators", range(30, 100, 20)),  # [0,1,2,3,4,5] -> [50,]
             "threshold": hp.choice("threshold", range(300, 500, 50))
             #"threshold": hp.randint("threshold", 400),
             }

trials = Trials()
best = fmin(optimize_rf, space, algo=tpe.suggest, max_evals=20, trials=trials)

att_message = [trials.trial_attachments(trial)['message'] for trial in trials.trials]
for score, para, misc in zip( trials.losses() ,
                              att_message,
                              [item.get('misc').get('vals') for item in trials.trials]
                              ):
    logger.debug(f'score:{"%9.6f"%score}, para:{para}, misc:{misc}')