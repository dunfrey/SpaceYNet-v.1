import config
import layers

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers, callbacks


def create_spaceynet_v1():
    input_data = Input((config.IMG_HEIGHT,
                        config.IMG_WIDTH,
                        config.IMG_CHANNEL))
    # DEPTH
    depth_downsampling = layers.depth_first_step(input_data)
    depth_upsampling = layers.depth_second_step(depth_downsampling)

    # POSE
    pose_network = layers.pose_second_path(depth_downsampling)
    pose_6_dof = layers.prepare_pose_output(pose_network)

    # NETWORK OUTPUTS
    cls_pose_xyz, cls_pose_wpqr = layers.output_pose(pose_6_dof)
    cls_depth = depth_upsampling

    model = network_output(input_data,
                           cls_pose_xyz,
                           cls_pose_wpqr,
                           cls_depth)

    return model


def create_spaceynet_v2():
    input_data = Input(batch_shape=(config.BATCH_SIZE,
                                    config.IMG_HEIGHT,
                                    config.IMG_WIDTH,
                                    config.IMG_CHANNEL))

    # DEPTH
    depth_left = layers.depth_first_step(input_data)
    depth_right = layers.depth_second_step(depth_left)

    # POSE
    pose_first_path = layers.pose_second_path(depth_left)
    pose_6_dof = layers.prepare_pose_output(pose_first_path)

    # LSTM
    pose_6_dof_lstm = layers.prepare_pose_output_lstm(pose_6_dof)

    # NETWORK OUTPUTS
    cls_pose_xyz, cls_pose_wpqr = layers.output_pose(pose_6_dof_lstm)
    cls_depth = depth_right

    model = network_output(input_data,
                           cls_pose_xyz,
                           cls_pose_wpqr,
                           cls_depth)

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
    adam = optimizers.Adam(lr=config.LEARNING_RATE,
                           decay=config.DECAY,
                           amsgrad=config.AMSGRAD)

    if depth_flag is True:
        losses = ['mae', 'mse', 'mse']
    else:
        losses = ['mse', 'mse']

    model.compile(optimizer=adam,
                  loss=losses,
                  metrics=['accuracy'])

    return model


def export_summary_model(path, model):
    with open(path + 'summary/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def extract_model(model_checkpoint_path):
    model = load_model(model_checkpoint_path)
    return model


def set_checkpointer(path):
    checkpoint_path = path + 'checkpoint/'
    checkpointer = \
        ModelCheckpoint(checkpoint_path + config.CK_CHOICE,
                        verbose=1,
                        save_best_only=True)
    return checkpointer


def set_reducer():
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_cls_position_loss',
        factor=0.9,
        verbose=1,
        patience=8,
        cooldown=3,
        min_lr=config.DECAY)
    return reduce_lr
