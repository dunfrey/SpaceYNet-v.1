import config
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


def depth_first_step(input_data):
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


def depth_second_step(input_data):
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
    conv = conv2d_downsampling(data, 16)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = conv2d_downsampling(conv, 16)
    return conv


def layer_depth_conv2d(data, filters):
    conv = conv2d_downsampling(data, filters)
    conv = conv2d_downsampling(conv, filters)
    return conv


def conv2d_downsampling(data, filters, strides=3):
    layer = Conv2D(filters,
                   (strides, strides),
                   kernel_initializer='he_normal',
                   padding='same')(data)
    return layer


def layer_depth_deconv(data, layer_to_concat, filters, axis=-1):
    deconv = deconv2d_upsampling(data, filters)
    concatenation = concatenate([deconv, layer_to_concat])
    if axis != -1:
        concatenation = concatenate([deconv,
                                     layer_to_concat],
                                    axis=axis)
    return concatenation


def deconv2d_upsampling(data, filters):
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
    cls_depth = Conv2D(config.IMG_CHANNEL,
                       (1, 1),
                       activation='sigmoid',
                       padding='same',
                       name='cls_depth')(data)
    return cls_depth


def pose_second_path(input_data):
    inception_1 = inception_layer(input_data,
                                  128, 128, 256, 32, 64, 3, 64)
    inception_2 = inception_layer(inception_1,
                                  112, 160, 288, 32, 64, 3, 64)
    inception_3 = inception_layer(inception_2,
                                  256, 160, 320, 32, 128, 3, 128)
    pool_1_3x3 = layer_maxpooling(inception_3,
                                  3,
                                  stride=2,
                                  padding='valid')
    inception_4 = inception_layer(pool_1_3x3,
                                  256, 160, 320, 32, 128, 3, 128)
    inception_5 = inception_layer(inception_4,
                                  384, 192, 384, 64, 128, 3, 128)
    return inception_5


def inception_layer(input_data,
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


def prepare_pose_output(data):
    layer_avgpool = layer_avgpooling(data, pool=3, stride=1)
    layer_flat = Flatten()(layer_avgpool)
    return layer_flat


def prepare_pose_output_lstm(data):
    target_tensor_size = (1, 1024)
    layer_reshape = Reshape(target_shape=target_tensor_size,
                            name='reshape_conv_lstm')(data)
    layer_lstm_1 = LSTM(256,
                        stateful=True,
                        return_sequences=True,
                        name='lstm_1')(layer_reshape)
    layer_lstm_2 = LSTM(config.BATCH_SIZE,
                        stateful=True,
                        name='lstm_2')(layer_lstm_1)
    return layer_lstm_2


def output_pose(data):
    layer_dropout = Dropout(.5)(data)
    cls_fc_position = Dense(2, name='cls_position')(layer_dropout)
    cls_fc_quaternion = Dense(2, name='cls_quaternion')(layer_dropout)
    return cls_fc_position, cls_fc_quaternion
