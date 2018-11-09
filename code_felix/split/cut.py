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
    outid_list = outid_list[outid_list.isin(get_score_outid())]
    num_sample = 'all_2'
else:
    outid_list = train.out_id.drop_duplicates()

    outid_list = outid_list[outid_list.isin(get_score_outid())].sample(num_sample)

    outid_list = list(outid_list.values)

    outid_list.extend(['673691705008931', '861181511175991', '861971709008361', ])

print(f'outid_len:{len(outid_list)}')
train = train[train.out_id.isin(outid_list)].sort_values('start_time', ascending=False)
print('Sort the train by start_time')
# st = pd.to_datetime(train.start_time)

# last_month = train[(st >= pd.to_datetime('2018-07-01') )& (st <= pd.to_datetime('2018-07-31'))]\

validate_list = []
i=0
for out_id in outid_list:
    i += 1
    if i%100==0:
        print(out_id, i, len(outid_list))
    mini = train.loc[train.out_id==out_id]
    validate_list.append(mini.head(10))

validate = pd.concat(validate_list)

train_train_file = f'{DATA_DIR}/train_{num_sample}.csv'
train_validate_file = f'{DATA_DIR}/test_{num_sample}.csv'



validate.to_csv(train_validate_file,index=None)
print(len(validate), len(validate.out_id.drop_duplicates()))

print(len(train))
train_train = train[~train.index.isin(validate.index)]
train_train.to_csv(train_train_file,index=None)
print(len(train_train), len(outid_list))