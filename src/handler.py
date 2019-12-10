import config as cfg
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def set_configurations(**kwargs):
    cfg.FOLDER_DATASET = kwargs['input_dataset_folder']
    cfg.OUTPUT_RESULT = kwargs['input_output_folder']
    cfg.CHECKPOINTER = kwargs['input_checkpoint_file']


def extract_text_file(**kwargs):
    cfg.log(extract_text_file.__module__,
            extract_text_file.__name__)
    if kwargs['input_file_data_train'] != '':
        with open(kwargs['input_file_data_train']) as f:
            for line in f:
                (image_name,
                 pos_x,
                 pos_y,
                 pos_z,
                 quat_w,
                 quat_p,
                 quat_q,
                 quat_r) = line.split()
                cfg.images_rgb.append(''.join(cfg.FOLDER_DATASET +
                                              image_name +
                                              '.jpg'))
                cfg.images_depth.append(''.join(cfg.FOLDER_DATASET +
                                                image_name.replace('rgb',
                                                                   'depth') +
                                                '.jpg'))
                cfg.images_label_axis.append((float(pos_x),
                                              float(pos_y),
                                              float(pos_z)))
                cfg.images_label_quat.append((float(quat_w),
                                             float(quat_p),
                                             float(quat_q),
                                             float(quat_r)))


def extract_data():
    cfg.log(extract_data.__module__,
            extract_data.__name__)
    cfg.images_label_axis, cfg.images_label_quat = \
        pose_standardization(cfg.images_label_axis, cfg.images_label_quat)
    cfg.images_rgb = extract_images(cfg.images_rgb)
    cfg.images_depth = extract_images(cfg.images_depth)


def pose_standardization(input_axis, input_quat):
    from sklearn.preprocessing import StandardScaler

    input_axis = np.asarray(input_axis)
    scaler_axis = StandardScaler().fit(input_axis)
    rescaled_axis = scaler_axis.transform(input_axis)

    input_quat = np.asarray(input_quat)
    scaler_quat = StandardScaler().fit(input_quat)
    rescaled_quat = scaler_quat.transform(input_quat)

    return rescaled_axis, rescaled_quat


def extract_images(input_images):
    import cv2
    images = np.zeros((len(input_images),
                       cfg.IMG_HEIGHT,
                       cfg.IMG_WIDTH,
                       cfg.IMG_CHANNEL),
                      dtype=np.uint8)
    for i in range(len(input_images)):
        image = cv2.imread(input_images[i])
        image = cv2.resize(image, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH))
        images[i] = image
    images = images.astype('float32')
    images /= 255
    return images


def fit():
    import models as mls
    import visualizer as vls
    model = mls.create_spaceynet()
    checkpointer = mls.create_checkpointer()
    stoper = mls.earlier_stop()
    output = model.fit(cfg.images_rgb,
                       [cfg.images_depth,
                        cfg.images_label_axis,
                        cfg.images_label_quat],
                       validation_split=cfg.VALID_SPLIT,
                       batch_size=cfg.BATCH_SIZE,
                       epochs=cfg.EPOCHS,
                       shuffle=True,
                       callbacks=[checkpointer,
                                  stoper])
    model.save(cfg.CHECKPOINTER)
    vls.loss_and_accuracy(output, stoper.stopped_epoch, 0)
