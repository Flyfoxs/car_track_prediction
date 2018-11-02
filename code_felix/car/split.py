from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
import pandas as pd
import numpy as np

import sys
if len(sys.argv)>1:
    num_sample =  int(sys.argv[1])
else:
    num_sample = 200
print("====num_sample:", num_sample)

train = pd.read_csv('./input/del/train_new.csv', delimiter=',', dtype=train_dict)
outid_list = train.out_id.drop_duplicates().sample(num_sample)

outid_list = list(outid_list.values)

outid_list.append('861181511175991')
outid_list.append('861971709008361')

print(f'outid_len:{len(outid_list)}')
train = train[train.out_id.isin(outid_list)]

st = pd.to_datetime(train.start_time)

validate = train[(st >= pd.to_datetime('2018-07-01') )& (st <= pd.to_datetime('2018-07-31'))].sample(len(outid_list)*10)
validate.to_csv(train_validate_file,index=None)
print(len(validate))

print(len(train))
train_train = train[~train.index.isin(validate.index)]
train_train.to_csv(train_train_file,index=None)
print(len(train_train))