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
    X.set_index('out_id', inplace=True)
    predict_input = X[get_feature_columns(feature_gp)]
    check_exception(predict_input)
    if isinstance(model, xgb.core.Booster):
        return model.predict(xgb.DMatrix(predict_input))
    elif isinstance(model, lgb.basic.Booster):
        #logger.debug(f'Xgboost:{type(model)}')
        return model.predict(predict_input)
    elif isinstance(model, RandomForestClassifier):
        return model.predict_proba(predict_input)


@file_cache(overwrite=True)
def gen_sub(sub, threshold, gp, model_type, **kw):
    args = locals()

    global feature_gp
    feature_gp = gp

    if sub == True:
        # Real get sub file
        cur_train = train_file
        cur_test = test_file
    else:
        # Validate file
        train_train_file = f'{DATA_DIR}/train_val_{sub}.csv'
        train_validate_file = f'{DATA_DIR}/test_val_{sub}.csv'
        cur_train = train_train_file
        cur_test = train_validate_file


    train = get_train_with_adjust_position(threshold, cur_train)
    train = analysis_start_zone_id(threshold, cur_train, train)

    test = get_test_with_adjust_position(threshold, cur_train, cur_test)
    test = analysis_start_zone_id(threshold, cur_train, test)


    test['predict_id'] = None
    test['predict_zone_id'] = None

    predict_list = []
    car_num = len(test.out_id.drop_duplicates())
    count = 0
    for out_id in test.out_id.drop_duplicates():
        count += 1
        classes_num = len(train[train.out_id == out_id].end_zoneid.drop_duplicates())
        if classes_num == 1:
            result = 0
        else:
            test_mini = test.loc[test.out_id == out_id]
            #logger.debug(f"Begin to train the model for car:{out_id}, records:{len(test_mini)}" )

            model = get_mode(out_id, train, model_type=model_type, **kw)
            result = predict(model, test_mini)
            #logger.debug(result.shape)
            result = np.argmax(result, axis=1)
        #logger.debug(result)

        test.loc[test.out_id == out_id, 'predict_id']  = result

        predict_result = get_zone_inf(out_id, train, test)
        sing_loss = cal_loss_for_df(predict_result)
        logger.debug(f"{count}/{car_num} loss:{'Sub model' if sing_loss is None else '{:,.4f}'.format(sing_loss)} "
                     f"for outid:{out_id}, num_cls:{classes_num}, {len(test.loc[test.out_id == out_id])} records, sub:{sub}")

        predict_list.append(predict_result)

    predict_list = pd.concat(predict_list)
    predict_list.set_index('r_key',inplace=True)

    #logger.debug(predict_list.head())

    #Reorder predict result
    predict_list = pd.DataFrame(index=test.r_key).join(predict_list)

    loss = cal_loss_for_df(predict_list)
    if loss:
        logger.debug(f"=====Loss is {'{:,.5f}'.format(loss)} on {car_num} cars, "
                     f"{len(get_feature_columns(feature_gp))} feature, "
                     f"{len(predict_list)} samples, args:{args}")


    sub_df = predict_list[['predict_lat', 'predict_lon']]
    sub_df.columns= ['end_lat','end_lon']
    sub_df.index.name = 'r_key'
    file_ensemble = f'./output/{threshold}/ensemble_rf_{args}.h5'
    save_df(predict_list, file_ensemble)
    if sub==True or loss is None:
        sub_file = replace_invalid_filename_char(f'./output/result_rf_{args}.csv')
        sub_df.to_csv(sub_file)
        logger.debug(f'Sub file is save to {sub_file}')

    return predict_list



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


