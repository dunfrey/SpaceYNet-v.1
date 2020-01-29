import config_file
import layers_nn

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers


def create_spaceynet_v1():
    input_data = Input((config_file.IMG_HEIGHT,
                        config_file.IMG_WIDTH,
                        config_file.IMG_CHANNEL))
    # DEPTH
    depth_left = layers_nn.unet_first_path(input_data)
    depth_right = layers_nn.unet_second_path(depth_left)

    # POSE
    pose_first_path = layers_nn.googlenet_second_path(depth_left)
    output_googlenet = \
        layers_nn.prepare_googlenet_output(pose_first_path)

    # TARGETS
    cls_pose_xyz, cls_pose_wpqr = layers_nn.output_pose(output_googlenet)
    cls_depth = depth_right

    model = network_output(input_data,
                           cls_pose_xyz,
                           cls_pose_wpqr,
                           cls_depth)

    return model


def create_spaceynet_v2():
    input_data = Input(batch_shape=(config_file.BATCH_SIZE,
                                    config_file.IMG_HEIGHT,
                                    config_file.IMG_WIDTH,
                                    config_file.IMG_CHANNEL))

    # DEPTH
    depth_left = layers_nn.unet_first_path(input_data)
    depth_right = layers_nn.unet_second_path(depth_left)

    # POSE
    pose_first_path = layers_nn.googlenet_second_path(depth_left)
    output_googlenet = \
        layers_nn.prepare_googlenet_output(pose_first_path)

    # LSTM
    output_googlenet_lstm = \
        layers_nn.prepare_googlenet_output_lstm(output_googlenet)

    # TARGETS
    cls_pose_xyz, cls_pose_wpqr = layers_nn.output_pose(output_googlenet_lstm)
    cls_depth = depth_right

    model = network_output(input_data,
                           cls_pose_xyz,
                           cls_pose_wpqr,
                           cls_depth)

    return model


def create_posenet():
    input_data = Input((config_file.IMG_HEIGHT,
                        config_file.IMG_WIDTH,
                        config_file.IMG_CHANNEL))
    # POSE
    pose_first_path = layers_nn.googlenet_first_path(input_data)

    pose_second_path = layers_nn.googlenet_second_path(pose_first_path)

    output_googlenet = \
        layers_nn.prepare_googlenet_output(pose_second_path)

    # TARGETS
    cls_pose_xyz, cls_pose_wpqr = layers_nn.output_pose(output_googlenet)

    model = network_output(input_data,
                           cls_pose_xyz,
                           cls_pose_wpqr,
                           None)

    return model


def create_contextualnet():
    input_data = Input(batch_shape=(config_file.BATCH_SIZE,
                                    config_file.IMG_HEIGHT,
                                    config_file.IMG_WIDTH,
                                    config_file.IMG_CHANNEL))

    # POSE
    pose_first_path = layers_nn.googlenet_first_path(input_data)

    pose_second_path = layers_nn.googlenet_second_path(pose_first_path)

    output_googlenet = \
        layers_nn.prepare_googlenet_output(pose_second_path)

    # LSTM
    output_googlenet_lstm = \
        layers_nn.prepare_googlenet_output_lstm(output_googlenet)

    # TARGETS
    cls_pose_xyz, cls_pose_wpqr = layers_nn.output_pose(output_googlenet_lstm)

    model = network_output(input_data,
                           cls_pose_xyz,
                           cls_pose_wpqr,
                           None)

    return model


def network_output(input_data,
                   cls_pose_xyz,
                   cls_pose_wpqr,
                   cls_depth=None):
    if cls_depth is not None:
        model = Model(inputs=[input_data],
                      outputs=[cls_depth,
                               cls_pose_xyz,
                               cls_pose_wpqr])
        model_output = set_model_params(model, True)

    else:
        model = Model(inputs=[input_data],
                      outputs=[cls_pose_xyz,
                               cls_pose_wpqr])
        model_output = set_model_params(model, False)

    return model_output


def set_model_params(model, depth_flag=False):
    adam = optimizers.Adam(lr=config_file.LEARNING_RATE,
                           decay=config_file.DECAY,
                           amsgrad=config_file.AMSGRAD)

    if depth_flag is True:
        losses = ['mae', 'mse', 'mse']
    else:
        losses = ['mse', 'mse']

    model.compile(optimizer=adam,
                  loss=losses,
                  metrics=['accuracy'])

    return model


def export_summary_model(path, model):
    with open(path + '/summary/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def extract_model(path, model):
    model = load_model(path + 'model_h5/' + model)
    return model
