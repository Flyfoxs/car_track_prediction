from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
import pandas as pd
import numpy as np
train = pd.read_csv(train_file, delimiter=',', dtype=train_dict)
outid_list = train.out_id.drop_duplicates()[:100]

train = train[train.out_id.isin(outid_list)]

st = pd.to_datetime(train.start_time)

validate = train[(st >= pd.to_datetime('2018-07-10') )& (st <= pd.to_datetime('2018-07-31'))].sample(len(outid_list)*10)
validate.to_csv(train_validate_file,index=None)
print(len(validate))

print(len(train))
train_train = train[~train.index.isin(validate.index)]
train_train.to_csv(train_train_file,index=None)
print(len(train_train))