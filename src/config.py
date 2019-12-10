import logging

logging.basicConfig(filename='log_file.log',
                    level=logging.INFO,
                    format='%(asctime)s :: %(message)s')


def log(file_name, function_name):
    logging.info(file_name + ' :: ' + function_name)


# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3
EPOCHS = 500
BATCH_SIZE = 65
VALID_SPLIT = 0.25
FOLDER_DATASET = []
OUTPUT_RESULT = []
CHECKPOINTER = []

images_label_axis = []
images_label_quat = []
images_rgb = []
images_depth = []
