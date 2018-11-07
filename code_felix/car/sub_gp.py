# 0.42434
from code_felix.car.train_rf import *
if __name__ == '__main__':
    #0.42434
    #gen_sub(200, 220, 0, max_depth=4)

    for threshold in [500,550, 450]:
        for sub in [ True ]:
            for feature_gp in range(0, 4):
                gen_sub(sub, threshold, feature_gp, max_depth=4)

    # gen_sub('all', 500, 2, max_depth=4)
    #gen_sub(True, 600, 2, max_depth=4)

