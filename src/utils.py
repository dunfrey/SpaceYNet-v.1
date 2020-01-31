import logging

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


def sizeof_train_validation_data(size_array):
    size_all = int(size_array / config.BATCH_SIZE)
    size_val = int((size_all * (config.VALID_SPLIT / 100)))
    size_train = size_all - size_val
    end_train = size_train * config.BATCH_SIZE
    size_all = size_all * config.BATCH_SIZE
    return end_train, size_all
