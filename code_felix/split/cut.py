from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
import pandas as pd
import numpy as np

import sys

from code_felix.split.group import get_worse_case

if len(sys.argv)>1:
    num_sample =  sys.argv[1]
else:
    num_sample = None
print("====num_sample:", num_sample)

train = pd.read_csv('./input/del/train_new.csv', delimiter=',', dtype=train_dict)
if num_sample is None:
    outid_list = train.out_id.drop_duplicates()
    outid_list = outid_list[outid_list.isin(get_score_outid().out_id)]
    num_sample = 'all_3'
elif num_sample =='worse2':
    # outid_list = train.out_id.drop_duplicates()
    outid_list =  get_worse_case(0.7, 5).out_id
    #num_sample = 'worse'
else:
    outid_list = train.out_id.drop_duplicates()

    outid_list = outid_list[outid_list.isin(get_score_outid().out_id)].sample(num_sample)

    outid_list = list(outid_list.values)

    outid_list.extend(['673691705008931', '861181511175991', '861971709008361', ])

logger.debug(f'outid_len:{len(outid_list)}')
train = train[train.out_id.isin(outid_list)].sort_values('start_time', ascending=False)
logger.debug('Sort the train by start_time')
# st = pd.to_datetime(train.start_time)

# last_month = train[(st >= pd.to_datetime('2018-07-01') )& (st <= pd.to_datetime('2018-07-31'))]\

validate_list = []
i=0
for out_id in outid_list:
    i += 1
    if i%100==0:
        logger.debug(f'{out_id}, {i}, {len(outid_list)}')
    mini = train.loc[train.out_id==out_id]
    count_val = len(mini)//5
    validate_list.append(mini.head(count_val))

validate = pd.concat(validate_list)

train_train_file = f'{DATA_DIR}/train_{num_sample}.csv'
train_validate_file = f'{DATA_DIR}/test_{num_sample}.csv'



validate.to_csv(train_validate_file,index=None)
#logger.debug(len(validate), len(validate.out_id.drop_duplicates()))

logger.debug(len(train))
train_train = train[~train.index.isin(validate.index)]
train_train.to_csv(train_train_file,index=None)
#logger.debug(len(train_train), len(outid_list))