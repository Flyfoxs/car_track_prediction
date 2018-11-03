from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
import pandas as pd
import numpy as np

import sys
if len(sys.argv)>1:
    num_sample =  int(sys.argv[1])
else:
    num_sample = None
print("====num_sample:", num_sample)

train = pd.read_csv('./input/del/train_new.csv', delimiter=',', dtype=train_dict)
if num_sample is None:
    outid_list = train.out_id.drop_duplicates()
else:
    outid_list = train.out_id.drop_duplicates().sample(num_sample)

    outid_list = list(outid_list.values)

    outid_list.append('861181511175991')
    outid_list.append('861971709008361')

print(f'outid_len:{len(outid_list)}')
train = train[train.out_id.isin(outid_list)]

st = pd.to_datetime(train.start_time)

last_month = train[(st >= pd.to_datetime('2018-07-01') )& (st <= pd.to_datetime('2018-07-31'))]\

validate_list = []
for out_id in outid_list:
    mini = last_month.loc[last_month.out_id==out_id]
    if len(mini)<=10:
        validate_list.append(mini)
    else:
        validate_list.append(mini.sample(10))
validate = pd.concat(validate_list)

validate.to_csv(train_validate_file,index=None)
print(len(validate))

print(len(train))
train_train = train[~train.index.isin(validate.index)]
train_train.to_csv(train_train_file,index=None)
print(len(train_train))