import networks
import config

import os.path
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config_tf = ConfigProto()
config_tf.gpu_options.allow_growth = True
session = InteractiveSession(config=config_tf)


def set_network(output_path):
    checkpoint_path = output_path + 'checkpoint/'
    with tf.device(config.DEVICE_WORK):

        # Checking if model not exits. If not, then create it
        if not os.path.exists(checkpoint_path + config.CK_CHOICE):
            model = charge_model_patch[config.CK_CHOICE]()

            networks.export_summary_model(output_path,
                                          model)

        else:
            model = networks.extract_model(checkpoint_path +
                                           config.CK_CHOICE)

    return model


def create_spaceynet_v1():
    return networks.create_spaceynet_v1()


def create_spaceynet_v2():
    return networks.create_spaceynet_v2()


charge_model_patch = {
    config.CK_SPACEYNET_V1: create_spaceynet_v1,
    config.CK_SPACEYNET_V2: create_spaceynet_v2
}
