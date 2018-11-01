
import logging
format_str = '%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s'
format = logging.Formatter(format_str)
logging.basicConfig(level=logging.DEBUG, format=format_str)

logger = logging.getLogger()

handler = logging.FileHandler('./log/forecast.log', 'a')
handler.setFormatter(format)
logger.addHandler(handler)



import functools
import time
def timed(logger=logger, level=None, show_begin=True, format='%s: %s ms', paras=True):
    if level is None:
        level = logging.DEBUG

    def need_show(item):
        import datetime as dt
        if isinstance(item, str) or isinstance(item, float) \
                or isinstance(item, int) or isinstance(item, dt.datetime):
            return True
        else:
            return False



    def _seq_but_not_str(obj):
        from collections import Iterable
        return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray))

    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            start = time.time()
            import pandas as pd
            args_mini = [item for item in args if need_show(item) ]

            kwargs_mini = [ (key, item) for key, item in kwargs.items() if need_show(item) ]

            if show_begin:
                if paras:
                    logger.info("Begin to run %s with:%r, %r" % (fn.__name__, args_mini, kwargs_mini))
                else:
                    logger.info(f"Begin to run {fn.__name__} with {len(args) + len(kwargs)} paras")
            result = fn(*args, **kwargs)
            result_mini = f'len:{len(result)}' if _seq_but_not_str(result) else result
            duration = time.time() - start
            logging.info('cost:%7.2f sec: ===%r end (%r, %r), result:%s ' % (duration, fn.__name__, args_mini, kwargs_mini, result_mini ))
            #logger.log(level, format, repr(fn), duration * 1000)
            return result
        return inner

    return decorator