import logging

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
