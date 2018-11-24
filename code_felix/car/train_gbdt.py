from sklearn import neighbors

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
#'start_base', 'distance','distance_min', 'distance_max', 'distance_mean',
# 'end_lat_adj', 'end_lon_adj','duration',  'end_lat', 'end_lon',  'day','start_lat_adj', 'start_lon_adj',
from code_felix.utils_.other import replace_invalid_filename_char, get_gpu_paras

feature_gp=0

def get_features(out_id, df):
    if df is None:
        return None, None

    if out_id is not None:
        df = df.loc[df.out_id == out_id]
    global feature_gp

    feature = df[get_feature_columns(feature_gp)]
    if 'start_lat' in feature:
        #disable SettingwithCopyWarning
        pd.reset_option('mode.chained_assignment')
        with pd.option_context('mode.chained_assignment', None):
            feature.start_lat = feature.start_lat.astype(float).values
            feature.start_lon = feature.start_lon.astype(float).values

    #logger.debug(f'feature_gp:{feature_gp}')
    if 'end_zoneid' in df:
        return  feature, df['end_zoneid'].astype('category')
    else:
        return feature, None

def train_model_lgb(X, Y,val_X, val_Y, **kw):
    num_round = kw['num_round']
    max_depth = kw['max_depth']
    min_data_in_leaf = kw['min_data_in_leaf']

    param = {'num_leaves': 31, 'verbose': -1,'max_depth': max_depth,
             'num_class': len(Y.cat.categories),
             'objective': 'multiclass',
             # 'boosting':'rf',
             # 'bagging_fraction':0.7,
             # 'bagging_freq':1,
             #  'feature_fraction':0.7,
             'min_data_in_leaf': min_data_in_leaf,
              #**get_gpu_paras('lgb')
             }
    param['metric'] = ['multi_logloss']

    #logger.debug(f'Final param for lgb is {param}')
    # 'num_leaves':num_leaves,

    train_data = lgb.Dataset(X, label=Y.cat.codes)
    test_data = lgb.Dataset(val_X, label=val_Y.cat.codes, reference=train_data)


    bst = lgb.train(param, train_data,
                    num_round,
                    early_stopping_rounds=20,
                    valid_sets=[test_data],
                    verbose_eval=False)

    #logger.debug(clf.feature_importances_)
    return bst
def train_model_rf(X, Y, **kw):
    clf = RandomForestClassifier(max_depth=kw['max_depth'],
                                 n_estimators = kw['num_round'],
                                 n_jobs=4,
                                 random_state=0)
    #Y = Y.astype('category')
    clf = clf.fit(X, Y.cat.codes)
    #logger.debug(clf.feature_importances_)
    return clf


def get_mode(out_id, df, df_val=None, model_type='lgb', **kw):
    if model_type == 'knn':
        global feature_gp
        feature_gp = 'knn'

    X, Y = get_features(out_id, df)
    val_X, val_Y = get_features(out_id, df_val)
    if model_type == 'lgb':
        model = train_model_lgb(X, Y,val_X, val_Y, **kw)
    elif model_type == 'rf':
        model = train_model_rf(X, Y, **kw)
    elif model_type == 'xgb':
        model = train_model_xgb(X, Y, **kw)
    elif model_type == 'knn':
        model = train_model_knn(X, Y, **kw)
    else:
        logger.error(f'Can not find model for {model_type}')
        exit(-1)

    return model


def train_model_xgb(X, Y, **kw):


    num_round = kw['num_round']
    max_depth = kw['max_depth']
    #logger.debug(f'{len(Y.cat.categories)}, {Y.cat.categories}')
    param = {'verbose': -1,'max_depth': max_depth,
             'num_class': len(Y.cat.categories),
             'boosting':'rf',
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


def train_model_knn(X, Y, **kw):
    n_neighbors= kw['n_neighbors'] if 'n_neighbors' in kw else 19 #15
    weights= kw['weights'] if 'weights' in kw else None #'distance'
    p=kw['p'] if 'p' in kw else 5
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, p=p)
    clf.fit(X, Y.cat.codes)


    return clf

def predict(model,  X):
    #global feature_gp
    # X.set_index('out_id', inplace=True)
    predict_input, _ = get_features(None, X)

    check_exception(predict_input)
    if isinstance(model, xgb.core.Booster):
        return model.predict(xgb.DMatrix(predict_input))
    elif isinstance(model, lgb.basic.Booster):
        #logger.debug(f'Xgboost:{type(model)}')
        return model.predict(predict_input, num_iteration=model.best_iteration)
    elif isinstance(model, RandomForestClassifier) \
            or isinstance(model, ExtraTreesClassifier)\
            or isinstance(model, KNeighborsClassifier)  :
        return model.predict_proba(predict_input)
    else:
        raise Exception(f'Unknown model:{model}')


@timed()
def gen_sub(file, threshold, gp, model_type, **kw):
    args = locals()
    logger_begin_paras(args)

    cur_train = f'{DATA_DIR}/train_{file}.csv'
    cur_test = f'{DATA_DIR}/test_{file}.csv'
    train = get_train_with_adjust_position(threshold, cur_train)
    test = get_test_with_adjust_position(threshold, cur_train, cur_test)

    out_id_list = test.out_id.drop_duplicates()

    train = train[train.out_id.isin(out_id_list)]
    test = test[test.out_id.isin(out_id_list)]

    sub, val, ensemble_test, ensemble_train = process_df(train, test, threshold, gp, model_type, **kw)

    loss, accuracy = cal_loss_for_df(val)

    file_ensemble = f'./output/ensemble/level1/{"{:,.5f}".format(loss or 0)}_{model_type}_gp{gp}_{threshold}_{args}.h5'
    save_df(val, sub, ensemble_test, ensemble_train, file_ensemble)

    logger.debug(f"=====Loss is {'{:,.5f}'.format(loss or 0)} on {len(out_id_list)} cars, "
                 f"{len(get_feature_columns(feature_gp))} feature, "
                 f"{len(val)} samples, args:{args}")


@timed()
def process_df(train, test, threshold, gp, model_type, **kw):
    split_num = kw['split_num']

    global feature_gp
    if model_type == 'knn':
        feature_gp = 'knn'
    else:
        feature_gp = gp


    # test = analysis_start_zone_id(threshold, cur_train, test)


    # test['predict_zone_id'] = None

    predict_list = []
    val_list = []
    ensemble_train = []
    ensemble_test = []
    car_num = len(test.out_id.drop_duplicates())
    count = 0
    for out_id in test.out_id.drop_duplicates(): #['861691702027751', '868260020674744'] : #
        count += 1
        single_test = test.loc[test.out_id == out_id].copy()
        single_train = train.loc[train.out_id == out_id].copy()
        #logger.debug(out_id)
        predict_result, val_result, test_result, train_result, message = predict_outid(kw, model_type, single_test, single_train, split_num)
        ensemble_train.append(train_result)
        ensemble_test.append(test_result)

        predict_result['model_type'] = model_type
        predict_result['threshold'] = threshold
        predict_result['kw'] = str(kw)
        logger.debug(f'{count}/{car_num}, {message}')

        predict_list.append(predict_result)
        if split_num > 1:
            val_result['model_type'] = model_type
            val_result['threshold'] = threshold
            val_result['kw'] = str(kw)
            val_list.append(val_result)
        else:
            val_list = None

    #logger.debug('=====1')
    predict_list = pd.concat(predict_list)
    predict_list.set_index('r_key',inplace=True)
    predict_list = pd.DataFrame(index=test.r_key).join(predict_list)
    if split_num > 1:
        val_list = pd.concat(val_list)
        val_list.set_index('r_key', inplace=True)
    else:
        val_list = pd.DataFrame()

    if ensemble_train is None or len(ensemble_train)==0:
        ensemble_train = [pd.DataFrame()]

    ensemble_test = pd.concat(ensemble_test)
    ensemble_train = pd.concat(ensemble_train)

    return predict_list, val_list, ensemble_test, ensemble_train

# def scale_df(train, test):
#
#     #logger.debug(f'Scale convert:{train.columns}')
#     scale = StandardScaler()
#     #logger.debug(f'{train.shape}, {test.shape}')
#     scale.fit(train[get_feature_columns('knn')])
#
#     col_list = get_feature_columns('knn')
#     train[col_list]  = scale.transform(train[col_list])
#     test[col_list] = scale.transform(test[col_list])
#     return  train, test

@timed()
def predict_outid(kw, model_type,  test, train, split_num =5):
    #logger.debug(len(train))
    out_id = train.out_id.values[0]


    test = test.sort_values('r_key')
    from sklearn.model_selection import KFold

    if split_num > 1:
        kf = KFold(n_splits=split_num, shuffle=True, random_state=777)
        split_partition = kf.split(train)
    else:
        split_partition = [(range(0, len(train)), None)]

    result_all = []
    val_all = []
    #split_partition = [(range(0, len(train)), 0)]
    for folder, (train_index, val_index) in enumerate(split_partition):
        #logger.debug('=====2')
        split_train = train.iloc[train_index]
        col_name = split_train.end_zoneid.astype('category').cat.categories
        if  val_index is not None and len(val_index)>0:
            split_val  = train.iloc[val_index]
        else:
            split_val = None
        #logger.debug(len(split_val))

        #logger.debug('=====3')
        classes_num = len(split_train.end_zoneid.drop_duplicates())
        if classes_num == 1:
            test_propability = np.ones((len(test), 1))
            val_propability = np.ones((len(split_val), 1))
        else:
            # logger.debug(f"Begin to train the model for car:{out_id}, records:{len(test_mini)}" )


            # train, test = scale_df(train, test)
            model = get_mode(out_id, split_train, split_val, model_type=model_type, **kw)
            test_propability = predict(model, test)
            if split_num > 1:
                val_propability = predict(model, split_val)

        #logger.debug('=====4')
        #result  = pd.DataFrame(result_propability, columns=col_name, index=test.r_key)
        test_propability = pd.DataFrame(test_propability, columns=col_name, index=test.r_key)
        test_propability.index.name = 'r_key'
        test_propability.reset_index(inplace=True)
        result_all.append(test_propability)
        #logger.debug('=====5')
        if split_num > 1:
            val_propability = pd.DataFrame(val_propability, columns=col_name, index=split_val.r_key)
            val_all.append(val_propability)

        # logger.debug(result[:10])
    logger.debug(f'Folder for outid#{out_id} is done')
    result_merge = pd.concat(result_all, ignore_index=True)
    test_result = result_merge.groupby('r_key').mean()
    #logger.debug(f'{result_merge.shape}, {test_propability.shape}' )
    logger.debug(f'End merge the result of {split_num} Kfolder')

    if split_num > 1:
        logger.debug(f'Try to analysis the validate set')
        train_result = pd.concat(val_all)
        val = train_result.idxmax(axis=1)
        train_val = train.copy(deep=True)
        train_val['predict_zone_id'] = val.loc[train_val.r_key].values
        val_result = get_zone_inf(out_id, train, train_val)
        val_loss, accuracy  = cal_loss_for_df(val_result)
        logger.debug(f'Val_loss:{val_loss}, accuracy:{accuracy} for out_id:{out_id}')
    else:
        #No split model
        val_result = pd.DataFrame()
        train_result = pd.DataFrame()


    result_zone_id = test_result.idxmax(axis=1)
    test['predict_zone_id'] = result_zone_id.values
    #logger.debug(test.predict_zone_id)
    predict_result = get_zone_inf(out_id, train, test)
    sing_loss, test_accuracy = cal_loss_for_df(predict_result)
    message = f"loss:{'Sub model' if sing_loss is None else '{:,.4f}'.format(sing_loss)}  " \
              f"outid:{out_id}, cls:{classes_num}, "\
              f"{len(test.loc[test.out_id == out_id])}/{len(train.loc[train.out_id == out_id])} records"
    return predict_result, val_result, test_result, train_result,  message

def filter_train(df, topn=50):
    out_id = df.out_id.values[0]
    end_count = df.groupby('end_zoneid')['end_zoneid'].count()

    end_count = end_count.sort_values(ascending=False)
    end_count_new = end_count.head(topn)
    logger.debug(f'Reduce the out:{out_id} end_zoneid from {len(end_count)} to  {len(end_count_new)}' )
    df = df[df.end_zoneid.isin(end_count_new.index)]
    return df




if __name__ == '__main__':
    #0.42434
    #gen_sub(200, 220, 0, max_depth=4)

    for num_round in range(10, 110, 10):
        gen_sub('100', 220, 0, 'xgb', max_depth=4, num_round=num_round)

    # for threshold in [500, 550, 450,]:
    #     for sub in [100, True, 'all']:
    #         for feature_gp in [1, 2, 3, 0, ]:
    #             gen_sub(sub, threshold, feature_gp, max_depth=4)
    #
    #


