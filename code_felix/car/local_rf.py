# 0.42434 + holiday =>0.42796
# 0.42434 + dis_center_0 => 0.42224
from code_felix.car.train_rf import *



for feature_gp in range(0,4):
    gen_sub(100, 220, feature_gp, max_depth=4)