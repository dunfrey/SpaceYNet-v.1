import config as cfg

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def create_spaceynet():

    def depth_network(input_layer):
        from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Activation
        c1 = convolution_2d_depth(16, 3, input_layer)
        c1 = BatchNormalization(name='init/norm1')(c1)
        c1 = Activation('relu')(c1)
        c1 = convolution_2d_depth(16, 3, c1)
        p1 = MaxPooling2D((2, 2))(c1)

        depth_layer_1 = convolution_2d_2x(32, p1)
        m1 = MaxPooling2D((2, 2))(depth_layer_1)
        depth_layer_2 = convolution_2d_2x(64, m1)
        m2 = MaxPooling2D((2, 2))(depth_layer_2)
        depth_layer_3 = convolution_2d_2x(128, m2)
        m3 = MaxPooling2D((2, 2))(depth_layer_3)

        c5 = convolution_2d_depth(256, 3, m3)
        c5 = convolution_2d_depth(256, 3, c5)
        d5 = Dropout(.5)(c5)

        u6 = convolution_transpose(128, d5)
        u6 = concatenate([u6, depth_layer_3])
        depth_layer_4 = convolution_2d_2x(128, u6)

        u7 = convolution_transpose(64, depth_layer_4)
        u7 = concatenate([u7, depth_layer_2])
        depth_layer_5 = convolution_2d_2x(64, u7)

        u8 = convolution_transpose(32, depth_layer_5)
        u8 = concatenate([u8, depth_layer_1])
        depth_layer_6 = convolution_2d_2x(32, u8)

        u9 = convolution_transpose(16, depth_layer_6)
        u9 = concatenate([u9, c1], axis=3)
        c9 = convolution_2d_depth(16, 3, u9)
        c9 = Conv2D(16, (3, 3), padding='same')(c9)
        return c9, d5

    def convolution_2d_depth(filters, stride, input_layer):
        from tensorflow.keras.layers import Conv2D
        layer = Conv2D(filters,
                       (stride, stride),
                       kernel_initializer='he_normal',
                       padding='same')(input_layer)
        return layer

    def convolution_2d_2x(filters, input_layer):
        c1 = convolution_2d_depth(filters, 3, input_layer)
        c1 = convolution_2d_depth(filters, 3, c1)
        return c1

    def inception_layer(first, sec, third, forth, fifth, sixth, seventh, input_layer):
        icp_1x1 = convolution_2d_pose(first, 1, input_layer)
        icp_3x3_reduce = convolution_2d_pose(sec, 1, input_layer)
        icp_3x3 = convolution_2d_pose(third, 3, icp_3x3_reduce)
        icp_5x5_reduce = convolution_2d_pose(forth, 1, input_layer)
        icp_5x5 = convolution_2d_pose(fifth, 5, icp_5x5_reduce)
        icp_pool = max_pooling_pose(sixth, 1, input_layer)
        icp_pool_proj = convolution_2d_pose(seventh, 1, icp_pool)
        icp_output = concatenate([icp_1x1,
                                  icp_3x3,
                                  icp_5x5,
                                  icp_pool_proj])
        return icp_output

    def convolution_transpose(filters, input_layer):
        from tensorflow.keras.layers import Conv2DTranspose
        layer = Conv2DTranspose(filters,
                                (2, 2),
                                strides=(2, 2),
                                padding='same')(input_layer)
        return layer

    def convolution_2d_pose(filters, stride, input_layer):
        from tensorflow.keras.layers import Conv2D
        layer = Conv2D(filters,
                       (stride, stride),
                       padding='same',
                       activation='relu')(input_layer)
        return layer

    def max_pooling_pose(pool, stride, input_layer):
        from tensorflow.keras.layers import MaxPooling2D
        layer = MaxPooling2D(pool_size=(pool, pool),
                             strides=(stride, stride),
                             padding='same')(input_layer)
        return layer

    with tf.device('/device:GPU:0'):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (Conv2D,
                                             concatenate,
                                             Input,
                                             AveragePooling2D,
                                             Flatten,
                                             Dense,
                                             Dropout)
        from tensorflow.keras import optimizers

        inputs = Input((cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNEL))
        depth_network, half_depth_network = depth_network(inputs)
        cls_depth = Conv2D(cfg.IMG_CHANNEL, (1, 1),
                           activation='sigmoid',
                           padding='same',
                           name='cls_depth')(depth_network)

        # first inception
        inception_1 = inception_layer(128, 128, 256, 24, 64, 3, 64,
                                      half_depth_network)
        inception_2 = inception_layer(112, 144, 288, 32, 64, 3, 64,
                                      inception_1)
        inception_3 = inception_layer(256, 160, 320, 32, 128, 3, 128,
                                      inception_2)
        pool_1_3x3 = max_pooling_pose(3, 2, inception_3)
        inception_4 = inception_layer(256, 160, 320, 32, 128, 3, 128,
                                      pool_1_3x3)
        inception_5 = inception_layer(384, 192, 384, 48, 128, 3, 128,
                                      inception_4)

        cls_pose = AveragePooling2D(pool_size=(3, 3),
                                    strides=(1, 1))(inception_5)
        cls_pose = Flatten()(cls_pose)
        cls_pose = Dropout(.5)(cls_pose)

        cls_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(cls_pose)
        cls_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(cls_pose)

        model_spaceynet = Model(inputs=[inputs],
                                outputs=[cls_depth,
                                         cls_pose_xyz,
                                         cls_pose_wpqr])

        with open(cfg.OUTPUT_RESULT + 'summary/modelSpaceYNet_summary.txt', 'w') as fh:
            model_spaceynet.summary(print_fn=lambda x: fh.write(x + '\n'))

        lr_ = 0.00315
        decay_ = lr_ / cfg.EPOCHS
        adam = optimizers.Adam(lr=lr_,
                               decay=decay_,
                               amsgrad=True)
        model_spaceynet.compile(optimizer=adam,
                                loss=['mae', 'mse', 'mse'],
                                metrics=['accuracy'])

    return model_spaceynet


def create_checkpointer():
    from tensorflow.keras.callbacks import ModelCheckpoint
    return ModelCheckpoint(cfg.CHECKPOINTER,
                           monitor='val_cls3_fc_pose_xyz_accuracy',
                           verbose=1,
                           save_best_only=True)


def earlier_stop():
    from tensorflow.keras.callbacks import EarlyStopping
    return EarlyStopping(monitor='val_cls3_fc_pose_xyz_loss',
                         patience=25)
