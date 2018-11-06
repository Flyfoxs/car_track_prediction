from code_felix.car.utils import *
from code_felix.car.distance_reduce import *
import pandas as pd
import numpy as np

def split_to_group(num=5, base_file='./'):
    """
    'start_zoneid', 'loss_dis', 'final_loss', 'out_id', 'predict_id', 'sn',  'predict_zone_id'

    zoneid, max(loss), min(loss), vag(loss)
    distinance, avg, max
    count(*), count(zoneid) in train, test

    :param num:
    :param base_file:
    :return:
    """
    pass