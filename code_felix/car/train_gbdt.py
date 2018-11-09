from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
#'start_base', 'distance','distance_min', 'distance_max', 'distance_mean',
# 'end_lat_adj', 'end_lon_adj','duration',  'end_lat', 'end_lon',  'day','start_lat_adj', 'start_lon_adj',
from code_felix.utils_.other import replace_invalid_filename_char, get_gpu_paras

feature_gp=0

def get_features(out_id, df):
    df = df[df.out_id == out_id]
    global feature_gp
    return df[get_feature_columns(feature_gp)] , df['end_zoneid'].astype('category')

def train_model_lgb(X, Y, **kw):
    num_round = kw['num_round']
    max_depth = kw['max_depth']

    param = {'num_leaves': 31, 'verbose': -1,'max_depth': max_depth,
             'num_class': len(Y.cat.categories),
             'objective': 'multiclass',
              #**get_gpu_paras('lgb')
             }
    param['metric'] = ['multi_logloss']

    #logger.debug(f'Final param for lgb is {param}')
    # 'num_leaves':num_leaves,

    train_data = lgb.Dataset(X, label=Y.cat.codes)
    test_data = lgb.Dataset(X, label=Y.cat.codes, reference=train_data)


    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data], verbose_eval=False)

    #logger.debug(clf.feature_importances_)
    return bst
def train_model_rf(X, Y, **kw):
    clf = RandomForestClassifier(max_depth=4, n_estimators = kw['num_round'], random_state=0)
    #Y = Y.astype('category')
    clf = clf.fit(X, Y.cat.codes)
    #logger.debug(clf.feature_importances_)
    return clf


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

def predict(model,  X):
    global feature_gp
    # X.set_index('out_id', inplace=True)
    predict_input = X[get_feature_columns(feature_gp)]
    check_exception(predict_input)
    if isinstance(model, xgb.core.Booster):
        return model.predict(xgb.DMatrix(predict_input))
    elif isinstance(model, lgb.basic.Booster):
        #logger.debug(f'Xgboost:{type(model)}')
        return model.predict(predict_input)
    elif isinstance(model, RandomForestClassifier):
        return model.predict_proba(predict_input)


@timed()
def gen_sub(file, threshold, gp, model_type, **kw):
    args = locals()
    cur_train = f'{DATA_DIR}/train_{file}.csv'
    cur_test = f'{DATA_DIR}/test_{file}.csv'

    train = get_train_with_adjust_position(threshold, cur_train)
    test = get_test_with_adjust_position(threshold, cur_train, cur_test)
    out_id_list = test.out_id

    val = process_df(train, test, threshold, gp, model_type, **kw)

    if file=='new':
        sub_df = val[['predict_lat', 'predict_lon']]
        sub_df.columns = ['end_lat', 'end_lon']
        sub_df.index.name = 'r_key'
        sub_file = replace_invalid_filename_char(f'./output/sub_{model_type}_{args}.csv')
        sub_df.to_csv(sub_file)
        logger.debug(f'Sub file is save to {sub_file}')
    else:
        loss = cal_loss_for_df(val)
        if loss:
            logger.debug(f"=====Loss is {'{:,.5f}'.format(loss)} on {len(out_id_list)} cars, "
                         f"{len(get_feature_columns(feature_gp))} feature, "
                         f"{len(val)} samples, args:{args}")


        cur_train = f'{DATA_DIR}/train_new.csv'
        cur_test = f'{DATA_DIR}/test_new.csv'
        train = get_train_with_adjust_position(threshold, cur_train)
        test = get_test_with_adjust_position(threshold, cur_train, cur_test)

        train = train[train.out_id.isin(out_id_list)]
        test = test[test.out_id.isin(out_id_list)]

        sub = process_df(train, test, threshold, gp, model_type, **kw)



        file_ensemble = f'./output/ensemble/{"{:,.3f}".format(loss)}_{model_type}_{threshold}_{args}.h5'
        save_df(val, sub, file_ensemble)


def process_df(train, test, threshold, gp, model_type, **kw):

    global feature_gp
    feature_gp = gp


    # test = analysis_start_zone_id(threshold, cur_train, test)


    # test['predict_zone_id'] = None

    predict_list = []
    car_num = len(test.out_id.drop_duplicates())
    count = 0
    for out_id in test.out_id.drop_duplicates():
        count += 1
        single_test = test.loc[test.out_id == out_id].copy()
        single_train = train.loc[train.out_id == out_id]

        predict_result, message = predict_outid(kw, model_type, single_test, single_train)
        predict_result['model_type'] = model_type
        predict_result['threshold'] = threshold
        predict_result['kw'] = str(kw)
        logger.debug(f'{count}/{car_num}, {message}')

        predict_list.append(predict_result)
        #

    predict_list = pd.concat(predict_list)
    predict_list.set_index('r_key',inplace=True)

    #logger.debug(predict_list.head())

    #Reorder predict result
    predict_list = pd.DataFrame(index=test.r_key).join(predict_list)




    return predict_list


def predict_outid(kw, model_type,  test, train):
    out_id = train.out_id.values[0]
    classes_num = len(train.end_zoneid.drop_duplicates())
    if classes_num == 1:
        result = 0
    else:
        # logger.debug(f"Begin to train the model for car:{out_id}, records:{len(test_mini)}" )

        model = get_mode(out_id, train, model_type=model_type, **kw)
        result = predict(model, test)
        # logger.debug(result.shape)
        result = np.argmax(result, axis=1)
    # logger.debug(result)
    test['predict_id'] = result
    predict_result = get_zone_inf(out_id, train, test)
    sing_loss = cal_loss_for_df(predict_result)
    message = f"loss:{'Sub model' if sing_loss is None else '{:,.4f}'.format(sing_loss)}  " \
              f"outid:{out_id}, cls:{classes_num}, "\
              f"{len(test.loc[test.out_id == out_id])}/{len(train.loc[train.out_id == out_id])} records"
    return predict_result, message


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


