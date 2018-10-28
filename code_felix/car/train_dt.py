from code_felix.car.utils import *
from code_felix.car.distance_reduce import *

def get_features(out_id, df):
    pass

def get_label(out_id, df):
    pass

def get_mode(out_id, df):
    pass

def predict(out_id, df):
    pass



if __name__ == '__main__':
    cur_train = train_train_file
    cur_test = train_validate_file
    threshold = 100
    train = get_train_with_adjust_position(100, cur_train)

    test = get_test_with_adjust_position(100, cur_train, cur_test)

