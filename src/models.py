import config as cfg
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout, Activation, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def create_spaceynet():
    # SPACEYNET
    with tf.device('/device:GPU:0'):
        inputs = Input((cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNEL))

        c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
        c1 = BatchNormalization(name='init/norm1')(c1)
        c1 = Activation('relu')(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        d5 = Dropout(.5)(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Conv2D(16, (3, 3), padding='same')(c9)

        cls_depth = Conv2D(cfg.IMG_CHANNEL, (1, 1),
                           activation='sigmoid',
                           padding='same',
                           name='cls_depth')(c9)

        input_pose = d5

        # inception 5
        icp_5_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp_5/1x1')(input_pose)
        icp_5_3x3_reduce = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp_5/3x3_reduce')(input_pose)
        icp_5_3x3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='icp_5/3x3')(icp_5_3x3_reduce)
        icp_5_5x5_reduce = Conv2D(24, (1, 1), padding='same', activation='relu', name='icp_5/5x5_reduce')(input_pose)
        icp_5_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='icp_5/5x5')(icp_5_5x5_reduce)
        icp_5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp_5/pool')(input_pose)
        icp_5_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp_5/pool_proj')(icp_5_pool)
        icp_5_output = concatenate([icp_5_1x1, icp_5_3x3, icp_5_5x5, icp_5_pool_proj], name='icp_5/output')

        # inception 6
        icp_6_1x1 = Conv2D(112, (1, 1), padding='same', activation='relu', name='icp_6/1x1')(icp_5_output)
        icp_6_3x3_reduce = Conv2D(144, (1, 1), padding='same', activation='relu', name='icp_6/3x3_reduce')(icp_5_output)
        icp_6_3x3 = Conv2D(288, (3, 3), padding='same', activation='relu', name='icp_6/3x3')(icp_6_3x3_reduce)
        icp_6_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp_6/5x5_reduce')(icp_5_output)
        icp_6_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='icp_6/5x5')(icp_6_5x5_reduce)
        icp_6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp_6/pool')(icp_5_output)
        icp_6_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp_6/pool_proj')(icp_6_pool)
        icp_6_output = concatenate([icp_6_1x1, icp_6_3x3, icp_6_5x5, icp_6_pool_proj], name='icp_6/output')

        # inception 7
        icp_7_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='icp_7/1x1')(icp_6_output)
        icp_7_3x3_reduce = Conv2D(160, (1, 1), padding='same', activation='relu', name='icp_7/3x3_reduce')(icp_6_output)
        icp_7_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='icp_7/3x3')(icp_7_3x3_reduce)
        icp_7_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp_7/5x5_reduce')(icp_6_output)
        icp_7_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='icp_7/5x5')(icp_7_5x5_reduce)
        icp_7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp_7/pool')(icp_6_output)
        icp_7_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp_7/pool_proj')(icp_7_pool)
        icp_7_output = concatenate([icp_7_1x1, icp_7_3x3, icp_7_5x5, icp_7_pool_proj], name='icp_7/output')

        pool3_3x3_icp8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool4/3x3_s2')(
            icp_7_output)
        # pool3_3x3_icp8 = Dropout(.1)(pool3_3x3_icp8)

        # inception 8
        icp_8_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='icp_8/1x1')(pool3_3x3_icp8)
        icp_8_3x3_reduce = Conv2D(160, (1, 1), padding='same', activation='relu', name='icp_8/3x3_reduce')(
            pool3_3x3_icp8)
        icp_8_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='icp_8/3x3')(icp_8_3x3_reduce)
        icp_8_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp_8/5x5_reduce')(
            pool3_3x3_icp8)
        icp_8_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='icp_8/5x5')(icp_8_5x5_reduce)
        icp_8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp_8/pool')(pool3_3x3_icp8)
        icp_8_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp_8/pool_proj')(icp_8_pool)
        icp_8_output = concatenate([icp_8_1x1, icp_8_3x3, icp_8_5x5, icp_8_pool_proj], name='icp_8/output')

        # inception 9
        icp_9_1x1 = Conv2D(384, (1, 1), padding='same', activation='relu', name='icp_9/1x1')(icp_8_output)
        icp_9_3x3_reduce = Conv2D(192, (1, 1), padding='same', activation='relu', name='icp_9/3x3_reduce')(icp_8_output)
        icp_9_3x3 = Conv2D(384, (3, 3), padding='same', activation='relu', name='icp_9/3x3')(icp_9_3x3_reduce)
        icp_9_5x5_reduce = Conv2D(48, (1, 1), padding='same', activation='relu', name='icp_9/5x5_reduce')(icp_8_output)
        icp_9_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='icp_9/5x5')(icp_9_5x5_reduce)
        icp_9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp_9/pool')(icp_8_output)
        icp_9_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp_9/pool_proj')(icp_9_pool)
        icp_9_output = concatenate([icp_9_1x1, icp_9_3x3, icp_9_5x5, icp_9_pool_proj], name='icp_9/output')

        cls_pose = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(icp_9_output)
        cls_pose = Flatten()(cls_pose)
        cls_pose = Dropout(.5)(cls_pose)
        cls_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(cls_pose)
        cls_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(cls_pose)

        model_spaceynet = Model(inputs=[inputs], outputs=[cls_depth, cls_pose_xyz, cls_pose_wpqr])

        # Print model summary to file
        with open(cfg.OUTPUT_RESULT + 'summary/modelSpaceYNet_summary.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model_spaceynet.summary(print_fn=lambda x: fh.write(x + '\n'))

        lr_ = 0.00315
        decay_ = lr_ / cfg.EPOCHS

        adam = optimizers.Adam(lr=lr_, amsgrad=True)
        model_spaceynet.compile(optimizer=adam, loss=['mae', 'mse', 'mse'], metrics=['accuracy'])

    return model_spaceynet


def create_checkpointer():
    return ModelCheckpoint(cfg.CHECKPOINTER,
                           monitor='val_cls3_fc_pose_xyz_accuracy',
                           verbose=1,
                           save_best_only=True)


def earlier_stop():
    return EarlyStopping(monitor='val_cls3_fc_pose_xyz_loss', patience=25)
