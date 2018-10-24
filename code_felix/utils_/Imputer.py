from collections import defaultdict

from sklearn.base import TransformerMixin
from code_felix.utils_.util_log import logger, timed
import numpy

class SeriesImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Object, then impute with the most frequent object.
        If the Series is not of dtype Object, then impute with the mean.

        """
        self.fill = None

    def fit(self, X, y=None):
        # logger.debug(X.name, X.dtype)
        # logger.debug("SeriesImputer:%s for %s:%s" % (id(self), X.name, X.dtype) )
        if X.name == 'renewed_yorn' :
            self.fill = 'null'
        elif 'datetime' in X.dtype.name :
            logger.warning("%s will be fill missing value with None as datetime" % X.name)
            self.fill = numpy.nan
        elif X.dtype.name in ['object', 'category']:
            if len(X.value_counts()) == 0:
                self.fill = numpy.nan
            else:
                self.fill = X.value_counts().index[0]
            logger.debug("Fill %s with value:%s, type:%s, count:%d"
                         % (X.name, self.fill, X.dtype, len(X.unique())))
        else:
            mean = X.mean()
            self.fill = 0 if numpy.math.isnan(mean)  else mean
            #self.fill = X.mean()
            logger.debug("Fill %s with value:%s, type:%s" % (X.name, self.fill, X.dtype))


        return self

    def transform(self, X, y=None):
        # if X.name in ('gap_avg', 'gap_pre'):
        #     logger.debug('No need to fill missing value for %s' % X.name)
        #     return X
        # else:
            logger.debug("Try to fill %s with value %s" % ( X.name, self.fill))
            X = X.replace([numpy.inf, -numpy.inf], numpy.nan)
            #X = X.apply(lambda x: self.fill if str(x).lower() in [' ', 'null', 'na'] else x)
            return X.fillna(self.fill)

    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X)



@timed()
def convert_missing(sample):
    imputer = defaultdict(SeriesImputer)
    train_temp = sample[sample.label_del == 'train']

    logger.debug("Begin to fill the missing data base on %s/%s" % (len(train_temp), len(sample)))

    logger.debug("try to get the fill value base on %d training data" % len(train_temp))
    train_temp = train_temp.apply(lambda x: imputer[x.name].fit(x) , reduce=False)

    del train_temp

    temp_list = sorted(imputer.items(), key = lambda item: sample[item[0]].dtype.name  )

    sample = sample.apply(lambda x: imputer[x.name].transform(x), reduce=False)
    logger.debug("There are %d columns already fill the missing value" % len(temp_list))


    logger.debug("End clean the data" + str(sample.shape))
    return sample