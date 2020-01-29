import config_file
import system_configurator

from tensorflow.keras.layers import (MaxPooling2D,
                                     AveragePooling2D,
                                     BatchNormalization,
                                     Activation,
                                     Conv2D,
                                     Dropout,
                                     Conv2DTranspose,
                                     concatenate,
                                     Flatten,
                                     Dense,
                                     Reshape,
                                     LSTM)

depth_concat_layers = []


def unet_first_path(input_data):
    initial_layer = input_data_processing_depth(input_data)
    max_pool = layer_maxpooling(initial_layer, 2)

    layer_depth_conv_1 = layer_depth_conv2d(max_pool, 32)
    layer_maxpool_1 = layer_maxpooling(layer_depth_conv_1, 2)

    layer_depth_conv_2 = layer_depth_conv2d(layer_maxpool_1, 64)
    layer_maxpool_2 = layer_maxpooling(layer_depth_conv_2, 2)

    layer_depth_conv_3 = layer_depth_conv2d(layer_maxpool_2, 128)
    layer_maxpool_3 = layer_maxpooling(layer_depth_conv_3, 2)

    layer_depth_conv_4 = layer_depth_conv2d(layer_maxpool_3, 256)
    layer_dropout = Dropout(.5)(layer_depth_conv_4)

    # To concatenate at the U-Net second path
    depth_concat_layers.append(layer_depth_conv_3)
    depth_concat_layers.append(layer_depth_conv_2)
    depth_concat_layers.append(layer_depth_conv_1)
    depth_concat_layers.append(initial_layer)

    return layer_dropout


def unet_second_path(input_data):
    layer_depth_deconv_1 = \
        layer_depth_deconv(input_data, depth_concat_layers[0], 128)

    layer_depth_conv_1 = \
        layer_depth_conv2d(layer_depth_deconv_1, 128)
    layer_depth_deconv_2 = \
        layer_depth_deconv(layer_depth_conv_1,
                           depth_concat_layers[1],
                           64)

    layer_depth_conv_2 = \
        layer_depth_conv2d(layer_depth_deconv_2, 64)
    layer_depth_deconv_3 = \
        layer_depth_deconv(layer_depth_conv_2,
                           depth_concat_layers[2],
                           32)

    layer_depth_conv_3 = \
        layer_depth_conv2d(layer_depth_deconv_3, 32)
    layer_depth_deconv_4 = \
        layer_depth_deconv(layer_depth_conv_3,
                           depth_concat_layers[3],
                           16,
                           axis=3)

    layer_depth_conv_4 = \
        layer_depth_conv2d(layer_depth_deconv_4, 16)

    output_data = depth_classification(layer_depth_conv_4)
    return output_data


def input_data_processing_depth(data):
    conv = conv2d_unet(data, 16)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = conv2d_unet(conv, 16)
    return conv


def layer_depth_conv2d(data, filters):
    conv = conv2d_unet(data, filters)
    conv = conv2d_unet(conv, filters)
    return conv


def conv2d_unet(data, filters, strides=3):
    layer = Conv2D(filters,
                   (strides, strides),
                   kernel_initializer='he_normal',
                   padding='same')(data)
    return layer


def layer_depth_deconv(data, layer_to_concat, filters, axis=-1):
    deconv = deconv_unet(data, filters)
    concatenation = concatenate([deconv, layer_to_concat])
    if axis != -1:
        concatenation = concatenate([deconv,
                                     layer_to_concat],
                                    axis=axis)
    return concatenation


def deconv_unet(data, filters):
    layer = Conv2DTranspose(filters,
                            (2, 2),
                            strides=(2, 2),
                            padding='same')(data)
    return layer


def layer_maxpooling(data, pool, stride=None, padding=None):
    if stride is not None and padding is not None:
        layer = MaxPooling2D(pool_size=(pool, pool),
                             strides=(stride, stride),
                             padding=padding)(data)
    elif stride is None:
        layer = MaxPooling2D(pool_size=(pool, pool),
                             padding='same')(data)
    elif stride is not None:
        layer = MaxPooling2D(pool_size=(pool, pool),
                             strides=(stride, stride),
                             padding='same')(data)
    else:
        layer = MaxPooling2D(pool_size=(pool, pool))(data)

    return layer


def depth_classification(data):
    cls_depth = Conv2D(config_file.IMG_CHANNEL,
                       (1, 1),
                       activation='sigmoid',
                       padding='same',
                       name='cls_depth')(data)
    return cls_depth


def googlenet_first_path(input_data):
    initial_layer = input_data_processing_pose(input_data)
    inception_1 = layer_inception(initial_layer,
                                  64, 96, 128, 16, 32, 3, 32)
    inception_2 = layer_inception(inception_1,
                                  128, 128, 192, 32, 96, 3, 64)
    pool_1_3x3 = layer_maxpooling(inception_2,
                                  3,
                                  stride=2,
                                  padding='valid')
    inception_3 = layer_inception(pool_1_3x3,
                                  192, 96, 208, 16, 48, 3, 64)
    inception_4 = layer_inception(inception_3,
                                  160, 112, 224, 24, 64, 3, 64)
    return inception_4


def input_data_processing_pose(input_data):
    layer_conv_1 = conv2d_googlenet(input_data, 64, 7)
    layer_maxpool_1 = layer_maxpooling(layer_conv_1, 3, 2, 'valid')
    batch_norm_1 = BatchNormalization()(layer_maxpool_1)
    layer_conv_2 = conv2d_googlenet(batch_norm_1, 64, 1)
    layer_conv_3 = conv2d_googlenet(layer_conv_2, 192, 3)
    batch_norm_2 = BatchNormalization()(layer_conv_3)
    layer_maxpool_2 = layer_maxpooling(batch_norm_2, 3, 2, 'valid')
    return layer_maxpool_2


def googlenet_second_path(input_data):
    inception_1 = layer_inception(input_data,
                                  128, 128, 256, 32, 64, 3, 64)
    inception_2 = layer_inception(inception_1,
                                  112, 160, 288, 32, 64, 3, 64)
    inception_3 = layer_inception(inception_2,
                                  256, 160, 320, 32, 128, 3, 128)
    pool_1_3x3 = layer_maxpooling(inception_3,
                                  3,
                                  stride=2,
                                  padding='valid')
    inception_4 = layer_inception(pool_1_3x3,
                                  256, 160, 320, 32, 128, 3, 128)
    inception_5 = layer_inception(inception_4,
                                  384, 192, 384, 64, 128, 3, 128)
    return inception_5


def layer_inception(input_data,
                    size_conv_1x1,
                    size_conv_3x3_red,
                    size_conv_3x3,
                    size_conv_5x5_red,
                    size_conv_5x5,
                    size_pool,
                    size_conv_1x1_red):
    icp_1x1 = conv2d_googlenet(input_data, size_conv_1x1)

    icp_3x3_reduce = conv2d_googlenet(input_data, size_conv_3x3_red)
    icp_3x3 = conv2d_googlenet(icp_3x3_reduce, size_conv_3x3, 3)

    icp_5x5_reduce = conv2d_googlenet(input_data, size_conv_5x5_red)
    icp_5x5 = conv2d_googlenet(icp_5x5_reduce, size_conv_5x5, 5)

    icp_pool = layer_maxpooling(input_data, size_pool, 1)
    icp_1x1_red = conv2d_googlenet(icp_pool, size_conv_1x1_red)

    concatenation = concatenate([icp_1x1,
                                 icp_3x3,
                                 icp_5x5,
                                 icp_1x1_red])
    return concatenation


def conv2d_googlenet(input_layer, filters, stride=1):
    layer = Conv2D(filters,
                   (stride, stride),
                   padding='same',
                   activation='relu')(input_layer)
    return layer


def layer_avgpooling(data, pool, stride=None):
    if stride is None:
        layer = AveragePooling2D(pool_size=(pool, pool))(data)
    else:
        layer = AveragePooling2D(pool_size=(pool, pool),
                                 strides=(stride, stride))(data)
    return layer


def prepare_googlenet_output(data):
    layer_avgpool = layer_avgpooling(data, pool=3, stride=1)
    layer_flat = Flatten()(layer_avgpool)
    return layer_flat


def prepare_googlenet_output_lstm(data):
    target_tensor_size = (1, 1024)
    layer_reshape = Reshape(target_shape=target_tensor_size,
                            name='reshape_conv_lstm')(data)
    layer_lstm_1 = LSTM(256,
                        stateful=True,
                        return_sequences=True,
                        name='lstm_1')(layer_reshape)
    layer_lstm_2 = LSTM(config_file.BATCH_SIZE,
                        stateful=True,
                        name='lstm_2')(layer_lstm_1)
    return layer_lstm_2


def output_pose(data):
    layer_dropout = Dropout(.5)(data)
    axis_size, quat_size = system_configurator.type_robot_pose()
    cls_fc_position = Dense(axis_size, name='cls_fc_position')(layer_dropout)
    cls_fc_quaternion = Dense(quat_size, name='cls_fc_quaternion')(layer_dropout)
    return cls_fc_position, cls_fc_quaternion
