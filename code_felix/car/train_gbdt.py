from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
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
    return df[feature_col] , df['end_zoneid'].astype('category')

def train_model_lgb(X, Y, **kw):
    # replace_map = Y.drop_duplicates().sort_values().reset_index(drop=True).to_frame()
    # #logger.debug(replace_map)
    # #logger.debug(type(Y))
    # Y.replace(dict(zip(replace_map.end_zoneid, replace_map.index)),  inplace=True)
    #
    # num_class = len(Y.drop_duplicates())
    # #logger.debug(f'num_class:{num_class}, len_sample:{len(X)}')



    param = {'num_leaves': 31, 'verbose': -1,'max_depth': 3,
             'num_class': len(Y.cat.categories),
             'objective': 'multiclass',
              #**get_gpu_paras('lgb')
             }
    param['metric'] = ['multi_logloss']

    #logger.debug(f'Final param for lgb is {param}')
    # 'num_leaves':num_leaves,

    train_data = lgb.Dataset(X, label=Y.cat.codes)
    test_data = lgb.Dataset(X, label=Y.cat.codes, reference=train_data)

    num_round = kw['num_round']
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data], verbose_eval=False)

    #logger.debug(clf.feature_importances_)
    return bst
def train_model_rf(X, Y, **kw):
    clf = RandomForestClassifier(max_depth=4, n_estimators = kw['num_round'], random_state=0)
    #Y = Y.astype('category')
    clf = clf.fit(X, Y.cat.codes)
    #logger.debug(clf.feature_importances_)
    return clf

def train_model_xgb(X, Y, **kw):


    num_round = kw['num_round']
    max_depth = kw['max_depth']
    #logger.debug(f'{len(Y.cat.categories)}, {Y.cat.categories}')
    param = {'verbose': -1,'max_depth': max_depth,
             'num_class': len(Y.cat.categories),
             'objective': 'multi:softprob',
             'silent': True,
              **get_gpu_paras('xgb')
             }
    param['eval_metric'] = ['mlogloss']

    #logger.debug(f'Final param for lgb is {param}')
    # 'num_leaves':num_leaves,

    train_data = xgb.DMatrix(X, Y.cat.codes)
    test_data = xgb.DMatrix(X, Y.cat.codes)


    bst = xgb.train(param, train_data, num_round, evals=[(test_data, 'train')], verbose_eval=False)

    #logger.debug(clf.feature_importances_)
    return bst

def get_mode(out_id, df, model_type='lgb', **kw):
    X, Y = get_features(out_id, df)
    if model_type == 'lgb':
        model = train_model_lgb(X, Y, **kw)
    elif model_type == 'rf':
        model = train_model_rf(X, Y, **kw)
    elif model_type == 'xgb':
        model = train_model_xgb(X, Y, **kw)
    else:
        logger.error(f'Can not find model for {model_type}')
        exit(-1)

    return model


def predict(model,  X):

    if isinstance(model, xgb.core.Booster):
        return model.predict(xgb.DMatrix(X[feature_col]))
    elif isinstance(model, lgb.basic.Booster):
        #logger.debug(f'Xgboost:{type(model)}')
        return model.predict(X[feature_col])
    elif isinstance(model, RandomForestClassifier):
        return model.predict_proba(X[feature_col])


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
    #model_type = 'xgb'
    if model_type in ['xgb', 'lgb', 'rf']:
        out_list = test.out_id.drop_duplicates()
        count=0
        for out_id in out_list:
            count += 1
            #logger.debug(f'Progress:{count}/{len(out_list)}')
            result = process_single_out_id(out_id, test, train, model_type, **kw, )
            logger.debug(f'{count}/{len(out_list)}, outid:{out_id}, {len(result)} records, {args}')

    # else:
    #     process_out_id = partial(process_single_out_id,  test=test, train=train, model_type=model_type, **kw, )   # (out_id, test, train)
    #     from multiprocessing.dummy import Pool as ThreadPool
    #     pool = ThreadPool(processes=4)
    #     results = pool.map(process_out_id, test.out_id.drop_duplicates())
    #     pool.close();
    #     pool.join()

    test = get_zone_inf(test, threshold)
    #logger.debug(test.head(1))


    #Reorder predict result
    #test = test.drop(test.columns, axis=1).join(test)

    #logger.debug(test.head(1))
    loss = cal_loss_for_df(test)
    if loss:
        logger.debug(f"Loss is {'{:,.5f}'.format(loss)}, on {len(test)} sample, args:{args}")
    if sub or loss is None:
        sub = test[['predict_lat', 'predict_lon']]
        sub.columns= ['end_lat','end_lon']
        sub.index.name = 'r_key'
        sub_file = replace_invalid_filename_char(f'./output/result_{model_type}_{args}.csv')
        sub.to_csv(sub_file)
        logger.debug(f'Sub file is save to {sub_file}')

    return test

#@timed(show_begin=False)
def process_single_out_id( out_id, test, train, model_type,**kw,):
    classes_num = len(train[train.out_id == out_id].end_zoneid.drop_duplicates())
    if classes_num == 1:
        test.loc[test.out_id == out_id, 'predict_id'] = 0
        test.loc[test.out_id == out_id, 'predict_zone_id'] = train[train.out_id == out_id].end_zoneid[0]
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
    Y = pd.Categorical(train.loc[train.out_id==out_id].end_zoneid).categories
    #zone_id = mini.loc.values
    #logger.debug(f'Convert {predict_id} to {zone_id}')
    return Y[predict_id]




if __name__ == '__main__':
    # for threshold in [400, 500, 600]:  # 1000,2000 ,300, 400, 500,
    #         for adjust_test in [False, True]:
    #             for max_depth in [8, 6, 4]:
    #                 for model_type in ['rf']:
    #                 #for estimator in range(50, 300, 50):
    #                     for num_round in range(8, 20, 2):
    #                         for sub in [False]:
    #                             gen_sub(sub, threshold, adjust_test,model_type, num_round = num_round, max_depth=max_depth )
    #                             exit(0)



    for max_depth in [4]:
        for sub in [False, ]:
            for adjust_test in [False, True]:
                for estimator in [ 10, 50, 100, 200]:
                    for threshold in[400,500,600]: #1000,2000 ,300, 400, 500,
                        for model_type in ['rf']:
                            gen_sub(sub, threshold, adjust_test,model_type, max_depth = max_depth, num_round=estimator,)
                            exit(0)
                            #exit(0)







