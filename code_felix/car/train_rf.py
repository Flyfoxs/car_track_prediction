from sklearn.ensemble import RandomForestClassifier

from code_felix.car.utils import get_feature_columns
from code_felix.car.distance_reduce import *
#'start_base', 'distance','distance_min', 'distance_max', 'distance_mean',
# 'end_lat_adj', 'end_lon_adj','duration',  'end_lat', 'end_lon',  'day','start_lat_adj', 'start_lon_adj',
from code_felix.utils_.other import replace_invalid_filename_char


topn=0

def get_features(out_id, df):
    df = df[df.out_id == out_id]
    global topn
    return get_feature_columns(df, topn) , df['end_zoneid'].astype('category')

def train_model(X, Y, **kw):
    estimate = kw['estimate'] if  'estimate' in kw else 100
    max_depth  = kw['max_depth'] if 'max_depth' in kw else 4
    clf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=0)

    clf = clf.fit(X, Y)
    #print(clf.feature_importances_)
    return clf

def get_mode(out_id, df, **kw):
    X, Y = get_features(out_id, df)
    model = train_model(X, Y, **kw)
    return model

def predict(model,  X):
    global topn
    return model.predict_proba(get_feature_columns(X,topn))


@file_cache(overwrite=True)
def gen_sub(sub, threshold, top_n, **kw):
    args = locals()

    global topn
    topn = top_n

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
    # if clean:
    #     train = clean_train_useless(train)

    test = get_test_with_adjust_position(threshold, cur_train, cur_test)

    # #topn=0
    # if topn>0:
    #     train  = cal_distance_2_centers(train, train_file, threshold, topn)
    #     logger.debug(f'Train columns: {train.columns}')
    #     test   = cal_distance_2_centers(test, train_file, threshold, topn)


    test['predict_id'] = None
    test['predict_zone_id'] = None

    predict_list = []
    car_num = len(test.out_id.drop_duplicates())
    count = 0
    for out_id in test.out_id.drop_duplicates():
        count += 1
        classes_num = len(train[train.out_id == out_id].end_zoneid.drop_duplicates())

        test_mini = test.loc[test.out_id == out_id]
        #logger.debug(f"Begin to train the model for car:{out_id}, records:{len(test_mini)}" )

        model = get_mode(out_id, train, **kw)
        result = predict(model, test_mini)
        #logger.debug(result.shape)
        result = np.argmax(result, axis=1)
        #logger.debug(result)

        test.loc[test.out_id == out_id, 'predict_id']  = result

        predict_result = get_zone_inf(out_id, train, test)
        sing_loss = cal_loss_for_df(predict_result)
        logger.debug(f"{count}/{car_num} loss:{'Sub model' if sing_loss is None else '{:,.4f}'.format(sing_loss)} "
                     f"for outid:{out_id}, {result.shape} records, sub:{sub}")

        predict_list.append(predict_result)

    predict_list = pd.concat(predict_list)
    predict_list.set_index('r_key',inplace=True)

    #logger.debug(predict_list.head())

    #Reorder predict result
    predict_list = pd.DataFrame(index=test.r_key).join(predict_list)

    loss = cal_loss_for_df(predict_list)
    if loss:
        logger.debug(f"=====Loss is {'{:,.5f}'.format(loss)} on {car_num} cars, {len(predict_list)} samples, args:{args}")


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

    for threshold in [220, 300, 400, 500, 600]:
        for sub in [100, True, 'all']:
            gen_sub(sub, threshold, 0, max_depth=4)


    # for topn in [4,5,7,0,]:
    #     for sub in [100]:
    #         gen_sub(sub, 220, topn, max_depth=4)

    #
    # for max_depth in [4, 5]:
    #     for threshold in [220, 300, 400, 500]:
    #         for sub in [True,False,]:
    #            # for clean in [True, False]:
    #                 gen_sub(sub, threshold, max_depth = max_depth)
    #                 exit(1)





