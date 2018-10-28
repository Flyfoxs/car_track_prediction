from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
import pandas as pd
import numpy as np
train = pd.read_csv(train_file, delimiter=',')

train.out_id = train.out_id.astype('str')

test = pd.read_csv(test_file, delimiter=',')

st = pd.to_datetime(train.start_time)

validate = train[(st >= pd.to_datetime('2018-07-01') )& (st <= pd.to_datetime('2018-07-31'))].sample(len(test))
validate.to_csv(train_validate_file,index=None)
print(len(validate))

print(len(train))
train_train = train[~train.index.isin(validate.index)]
train_train.to_csv(train_train_file,index=None)
print(len(train_train))