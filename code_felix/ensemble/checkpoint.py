from keras.callbacks import Callback
import numpy as np
from code_felix.utils_.util_log import *

class ReviewCheckpoint(Callback):

    def __init__(self, folder, val_train, val_label ):
        super(ReviewCheckpoint, self).__init__()
        logger.debug(f'val_train:{val_train.shape}, val_label:{val_label.shape} ')
        self.val_train = val_train
        self.val_label = val_label
        self.accuracy = 0
        self.folder = folder



    def on_epoch_end(self, epoch, logs=None):
        filepath = './output/model/checkpoint.h5'
        val_res = self.model.predict(self.val_train)
        acc = np.mean(np.argmax(val_res, axis=1) == np.argmax(self.val_label.values, axis=1))
        if acc > self.accuracy:
            self.accuracy = acc
            self.model.save(filepath, overwrite=True)
            logger.debug(f'Folder:{self.folder}, accuracy is:{acc}, epoch:{epoch}')



if __name__ == '__main__':
    pass
    # model = Inception_tranfer(-1, unlock=0).gen_model()
    # show_img_with_tags(df_test, -1, 224, model, random=False)


