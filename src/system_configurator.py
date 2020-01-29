import models_nn
import config_file

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def set_network(method, main_folder):
    neural_network_model = []
    with tf.device('/device:GPU:0'):
        if method == 'train':
            if config_file.CK_CHOICE == config_file.CK_SPACEYNET_V1:
                neural_network_model = models_nn.create_spaceynet_v1()
            elif config_file.CK_CHOICE == config_file.CK_SPACEYNET_V2:
                neural_network_model = models_nn.create_spaceynet_v2()
            elif config_file.CK_CHOICE == config_file.CK_POSENET:
                neural_network_model = models_nn.create_posenet()
            elif config_file.CK_CHOICE == config_file.CK_CONTEXTUALNET:
                neural_network_model = models_nn.create_contextualnet()

            models_nn.export_summary_model(main_folder,
                                           neural_network_model)

        if method == 'test':
            neural_network_model = \
                models_nn.extract_model(main_folder,
                                        config_file.CK_CHOICE)

    return neural_network_model


def create_checkpointer(path):
    checkpointer = \
        ModelCheckpoint(path + 'model_h5/' + config_file.CK_CHOICE,
                        verbose=1,
                        save_best_only=True)
    return checkpointer


def create_reducer():
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_cls_pose_xyz_loss',
        factor=0.9,
        verbose=1,
        patience=8,
        cooldown=3,
        min_lr=0.00000146)
    return reduce_lr


def type_robot_pose():
    if config_file.ROBOT_TYPE == 0:
        return 2, 2
    if config_file.ROBOT_TYPE == 1:
        return 3, 4
