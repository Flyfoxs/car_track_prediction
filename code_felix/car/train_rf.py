from sklearn.ensemble import RandomForestClassifier

from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
#'start_base', 'distance','distance_min', 'distance_max', 'distance_mean',
# 'end_lat_adj', 'end_lon_adj','duration',  'end_lat', 'end_lon',  'day','start_lat_adj', 'start_lon_adj',
from code_felix.utils_.other import replace_invalid_filename_char

feature_col = ['weekday', 'weekend', #'weekday',
               #'holiday',
               'hour','start_zoneid', ]

def get_features(out_id, df):
    df = df[df.out_id == out_id]
    #dup_label =  df['end_zoneid'].drop_duplicates()
    #logger.debug(f'Label:from {dup_label.min()} to {dup_label.max()}, length:{len(dup_label)} ')
    return df[feature_col] , df['end_zoneid']

def train_model(X, Y, **kw):
    #logger.debug(f'RandomForestClassifier:{kw}')
    clf = RandomForestClassifier( **kw, random_state=0)

    clf = clf.fit(X, Y)
    #logger.debug(clf.feature_importances_)
    return clf

def get_mode(out_id, df, **kw):
    X, Y = get_features(out_id, df)
    model = train_model(X, Y, **kw)
    return model

def predict(model,  X):
    #X = df[df.out_id==out_id]
    return model.predict_proba(X[feature_col])


@file_cache(overwrite=True)
@timed()
def gen_sub(sub, threshold, topn, **kw):
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
    if topn > 0:
        train = cal_distance_2_centers(train, cur_train, threshold, topn)
        test = cal_distance_2_centers(test, cur_train, threshold, topn)
    #adjust_test = False
    # if adjust_test:
    #     test = adjust_new_zoneid_in_test(threshold, test, cur_train)

    #Prepare null column to save predict result
    test['predict_id'] = None
    test['predict_zone_id'] = None
    # predict_cols = ['predict_id','predict_zone_id', 'predict_lat', 'predict_lon']
    # test = pd.concat([test, pd.DataFrame(columns=predict_cols)])

    car_num = len(test.out_id.drop_duplicates())
    count = 0
    for out_id in test.out_id.drop_duplicates():
        count += 1
        classes_num = len(train[train.out_id == out_id].end_zoneid.drop_duplicates())
        model = get_mode(out_id, train, **kw)
        result = predict(model, test.loc[test.out_id == out_id])
        #logger.debug(f'out_id:{out_id}, {result.shape}, raw_result:{result}')
        #logger.debug(result.shape)
        predict_id = np.argmax(result, axis=1)

        test.loc[test.out_id == out_id, 'predict_id'] = predict_id


        predict_zoneid = get_zone_id(predict_id, train, out_id)

        test.loc[test.out_id == out_id, 'predict_zone_id'] = predict_zoneid

        logger.debug(f'{count}/{car_num} predict for outid:{out_id}, {result.shape} records, sub:{sub}')


    test = get_zone_inf( test, threshold)


    #Reorder predict result
    #test = test.drop(test.columns, axis=1).join(test)


    loss = cal_loss_for_df(test)
    if loss:
        logger.debug(f"Loss is {'{:,.5f}'.format(loss)}, args:{args}")
    if sub or loss is None:
        sub = test[['predict_lat', 'predict_lon']]
        sub.columns= ['end_lat','end_lon']
        sub.index.name = 'r_key'
        sub_file = replace_invalid_filename_char(f'./output/result_rf_{args}.csv')
        sub.to_csv(sub_file)
        logger.debug(f'Sub file is save to {sub_file}')


    return test

def get_zone_id(predict_id, train, out_id):
    cat = pd.Categorical(train.loc[train.out_id == out_id].end_zoneid).categories
    #logger.debug(f'Convert {predict_id} to {zone_id}')
    return cat[predict_id]

if __name__ == '__main__':
    #gen_sub(False, 500, max_depth=4, n_estimators=20, )

    for sub in [False, True]:
        for max_depth in [4]:
            #for adjust_test in [0, 1000, 2000, 3000, 4000, 10000]:
                for estimator in [20]:
                    #for add_feature in ['weekday', 'holiday', None]:
                        for threshold in[500]: #1000,2000 ,300, 400, 500,
                            for topn in [3, 1, 5, 0, 10 ]:
                                gen_sub(sub, threshold, topn, max_depth = max_depth, n_estimators=estimator,)






