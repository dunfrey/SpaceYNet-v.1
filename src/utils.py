import logging
import math

import config

logging.basicConfig(filename='log_file.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s :: %(message)s')


def save_log(message):
    """Logs a message with level INFO on the root logger.
    Args:
        message (str): message describing some fact to save.
    """
    logging.info(message)


def split_train_validation(size_array):
    size_all = int(size_array / config.BATCH_SIZE)
    size_val = int((size_all * (config.VALID_SPLIT / 100)))
    size_train = size_all - size_val
    end_train = size_train * config.BATCH_SIZE
    size_all = size_all * config.BATCH_SIZE
    return end_train, size_all


def quaternion_to_euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]
