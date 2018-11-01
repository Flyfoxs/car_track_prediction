from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
#'start_base', 'distance','distance_min', 'distance_max', 'distance_mean',
# 'end_lat_adj', 'end_lon_adj','duration',  'end_lat', 'end_lon',  'day','start_lat_adj', 'start_lon_adj',
from code_felix.utils_.other import replace_invalid_filename_char, get_gpu_paras

feature_col = ['weekday', 'weekend', #'weekday',
               #'holiday',
               'hour','start_zoneid', ]

def get_features(out_id, df):
    df = df[df.out_id == out_id]
    #dup_label =  df['end_zoneid'].drop_duplicates()
    #logger.debug(f'Label:from {dup_label.min()} to {dup_label.max()}, length:{len(dup_label)} ')
    return df[feature_col] , df['end_zoneid']

def train_model_lgb(X, Y, **kw):
    replace_map = Y.drop_duplicates().sort_values().reset_index(drop=True).to_frame()
    #logger.debug(replace_map)
    #logger.debug(type(Y))
    Y.replace(dict(zip(replace_map.end_zoneid, replace_map.index)),  inplace=True)

    num_class = len(Y.drop_duplicates())
    #logger.debug(f'num_class:{num_class}, len_sample:{len(X)}')


    import lightgbm as lgb
    param = {'num_leaves': 31, 'verbose': -1,'max_depth': 3,
             'num_class': num_class,
             'objective': 'multiclass',
              #**get_gpu_paras('lgb')
             }
    param['metric'] = ['multi_logloss']

    #logger.debug(f'Final param for lgb is {param}')
    # 'num_leaves':num_leaves,

    train_data = lgb.Dataset(X, label=Y)
    test_data = lgb.Dataset(X, label=Y, reference=train_data)

    num_round = kw['num_round']
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data], verbose_eval=False)

    #logger.debug(clf.feature_importances_)
    return bst


def train_model_xgb(X, Y, **kw):
    replace_map = Y.drop_duplicates().sort_values().reset_index(drop=True).to_frame()
    #logger.debug(replace_map)
    #logger.debug(type(Y))
    Y.replace(dict(zip(replace_map.end_zoneid, replace_map.index)),  inplace=True)

    num_class = len(Y.drop_duplicates())
    #logger.debug(f'num_class:{num_class}, len_sample:{len(X)}')




    param = {'num_leaves': 31, 'verbose': -1,'max_depth': 3,
             'num_class': num_class,
             'objective': 'multi:softprob',
             'silent': True,
              **get_gpu_paras('xgb')
             }
    param['eval_metric'] = ['mlogloss']

    #logger.debug(f'Final param for lgb is {param}')
    # 'num_leaves':num_leaves,

    train_data = xgb.DMatrix(X, Y)
    test_data = xgb.DMatrix(X, Y)

    num_round = kw['num_round']
    bst = xgb.train(param, train_data, num_round, evals=[(test_data, 'train')], verbose_eval=False)

    #logger.debug(clf.feature_importances_)
    return bst

def get_mode(out_id, df, model_type='lgb', **kw):
    X, Y = get_features(out_id, df)
    if model_type == 'lgb':
        model = train_model_lgb(X, Y, **kw)
    else:
        model = train_model_xgb(X, Y, **kw)

    return model


def predict(model,  X):
    #logger.debug(type(model))
    if isinstance(model, xgb.core.Booster):
        return model.predict(xgb.DMatrix(X[feature_col]))
    else: #Lgb
        return model.predict(X[feature_col])


#@file_cache(overwrite=True)
@timed()
def gen_sub(sub, threshold, adjust_test,model_type, **kw):
    args = locals()


    if sub:
        # Real get sub file
        cur_train = train_file
        cur_test = test_file
    else:
        # Validate file
        cur_train = train_train_file
        cur_test = train_validate_file


    train = get_train_with_adjust_position(threshold, cur_train)
    # if clean:
    #     train = clean_train_useless(train)

    test = get_test_with_adjust_position(threshold, cur_train, cur_test)
    #logger.debug(test.head(1))
    if adjust_test:
        test = adjust_new_zoneid_in_test(threshold, test, cur_train)

    #Prepare null column to save predict result
    test['predict_id'] = None
    test['predict_zone_id'] = None
    # predict_cols = ['predict_id','predict_zone_id', 'predict_lat', 'predict_lon']
    # test = pd.concat([test, pd.DataFrame(columns=predict_cols)])

    # for out_id in test.out_id.drop_duplicates():
    #     #logger.debug(f"Begin to train the model for car:{out_id}" )
    model_type = 'xgb'

    process_out_id = partial(process_single_out_id,  test=test, train=train, model_type=model_type, **kw, )   # (out_id, test, train)

    from multiprocessing.dummy import Pool as ThreadPool
    if model_type == 'xgb':
        thread_num=1
    else:
        thread_num=4
    pool = ThreadPool(processes=thread_num)
    results = pool.map(process_out_id, test.out_id.drop_duplicates())
    pool.close();
    pool.join()

    test = get_zone_inf(test, threshold)
    #logger.debug(test.head(1))


    #Reorder predict result
    #test = test.drop(test.columns, axis=1).join(test)


    loss = cal_loss_for_df(test)
    if loss:
        logger.debug(f"Loss is {loss}, LGB args:{args}")
    if sub or loss is None:
        sub = test[['predict_lat', 'predict_lon']]
        sub.columns= ['end_lat','end_lon']
        sub.index.name = 'r_key'
        sub_file = replace_invalid_filename_char(f'./output/result_{model_type}_{args}.csv')
        sub.to_csv(sub_file)
        logger.debug(f'Sub file is save to {sub_file}')

    return test

@timed(show_begin=False)
def process_single_out_id( out_id, test, train, model_type,**kw,):
    classes_num = len(train[train.out_id == out_id].end_zoneid.drop_duplicates())
    if classes_num == 1:
        test.loc[test.out_id == out_id, 'predict_zone_id'] = 0
        test.loc[test.out_id == out_id, 'predict_id'] = train[train.out_id == out_id].end_zoneid[0]
        logger.debug(f'Finish the predict(simple way) for outid:{out_id}, {len(test.loc[test.out_id == out_id])} records')
    else:
        model = get_mode(out_id, train, model_type, **kw)
        result = predict(model, test.loc[test.out_id == out_id])
        # logger.debug(f'out_id:{out_id}, {result.shape}, raw_result:{result}')
        # logger.debug(result.shape)
        predict_id = np.argmax(result, axis=1)
        test.loc[test.out_id == out_id, 'predict_id'] = predict_id

        # logger.debug(f'out_id:{out_id}, ')
        # if out_id == '861181511140011':
        #     :(result)
        predict_zoneid = get_zone_id(predict_id, train, out_id)

        test.loc[test.out_id == out_id, 'predict_zone_id'] = predict_zoneid

        #logger.debug(f'Done predict outid:{out_id}, {result.shape} records, {threshold}, {sub}')

    return test[test.out_id==out_id]


def get_zone_id(predict_id, train, out_id):
    mini = train.loc[train.out_id==out_id]
    mini = mini.end_zoneid.sort_values().drop_duplicates()
    mini = mini.reset_index(drop=True)
    zone_id = mini.loc[predict_id].values
    #logger.debug(f'Convert {predict_id} to {zone_id}')
    return zone_id




if __name__ == '__main__':
    for threshold in [400, 500, 600]:  # 1000,2000 ,300, 400, 500,
            for adjust_test in [False, True]:
                for model_type in ['xgb', 'lgb']:
                #for estimator in range(50, 300, 50):
                    for num_round in [50, 100]:
                        for sub in [True]:
                            gen_sub(sub, threshold, adjust_test,model_type, num_round = num_round, )
                            ##exit(0)






