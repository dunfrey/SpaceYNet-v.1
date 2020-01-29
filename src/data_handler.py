import utils
import config_file
import random
import numpy

from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2


def shuffle_data(x_img, y_depth, y_axis, y_quat):
    data = list(zip(x_img, y_depth, y_axis, y_quat))
    random.shuffle(data)
    x_img, y_depth, y_axis, y_quat = zip(*data)
    return x_img, y_depth, y_axis, y_quat


def split_train_val_data(x_img, y_depth, y_axis, y_quat):
    size_all = int(len(x_img) / config_file.BATCH_SIZE)
    size_val = int((size_all * (config_file.VALID_SPLIT/100)))
    size_train = size_all - size_val

    X_img, y_depth, y_axis, y_quat = \
        shuffle_data(x_img, y_depth, y_axis, y_quat)

    end_train = size_train * 65
    size_all = size_all * 65

    X_img_train = numpy.array(X_img)[:end_train]
    y_depth_train = numpy.array(y_depth)[:end_train]
    y_axis_train = numpy.array(y_axis)[:end_train]
    y_quat_train = numpy.array(y_quat)[:end_train]
    X_img_val = numpy.array(X_img)[end_train:size_all]
    y_depth_val = numpy.array(y_depth)[end_train:size_all]
    y_axis_val = numpy.array(y_axis)[end_train:size_all]
    y_quat_val = numpy.array(y_quat)[end_train:size_all]

    return X_img_train, \
        y_depth_train, \
        y_axis_train, \
        y_quat_train, \
        X_img_val, \
        y_depth_val, \
        y_axis_val, \
        y_quat_val


def extract_data(method, path):
    utils.save_log(f'{extract_data.__module__} :: '
                   f'{extract_data.__name__}')

    print('\n\n----- READING DATA. PLEASE, WAIT A MINUTE -----\n\n')

    images_rgb, images_depth, labels_axis, labels_quat = \
        extract_data_from_file(method, path)

    labels_axis, labels_quat = \
        pose_standardization(labels_axis, labels_quat)

    images_rgb = extract_images(images_rgb)
    images_depth = extract_images(images_depth)

    return images_rgb, \
        images_depth, \
        labels_axis, \
        labels_quat


def extract_images(input_images):
    utils.save_log(f'{extract_images.__module__} :: '
                   f'{extract_images.__name__}')
    images = np.zeros((len(input_images),
                       config_file.IMG_HEIGHT,
                       config_file.IMG_WIDTH,
                       config_file.IMG_CHANNEL),
                      dtype=np.uint8)
    for i in range(len(input_images)):
        image = cv2.imread(input_images[i])
        image = cv2.resize(image, (config_file.IMG_HEIGHT,
                                   config_file.IMG_WIDTH))
        images[i] = image
    images = images.astype('float32')
    images /= 255
    return images


def pose_standardization(input_axis, input_quat):
    utils.save_log(f'{pose_standardization.__module__} :: '
                   f'{pose_standardization.__name__}')
    input_axis = np.asarray(input_axis)
    scaler_axis = StandardScaler().fit(input_axis)
    rescaled_axis = scaler_axis.transform(input_axis)

    input_quat = np.asarray(input_quat)
    scaler_quat = StandardScaler().fit(input_quat)
    rescaled_quat = scaler_quat.transform(input_quat)

    return rescaled_axis, rescaled_quat


def extract_data_from_file(method, path):
    utils.save_log(f'{extract_data_from_file.__module__} :: '
                   f'{extract_data_from_file.__name__}')

    if method == 'train':
        input_file = 'dataset_train.txt'
    if method == 'test':
        input_file = 'dataset_test.txt'

    images_rgb = []
    images_depth = []
    images_label_axis = []
    images_label_quat = []

    # axis_size, quat_size = system_configurator.type_robot_pose()
    # pos = ['pos_{}'.format(i) for i in range(0, axis_size)]
    # quat = ['quat_{}'.format(i) for i in range(0, axis_size)]

    with open(path + input_file) as f:
        for line in f:
            (image_name,
             pos_x,
             pos_y,
             # pos_z,
             quat_w,
             # quat_p,
             # quat_q,
             quat_r) = line.split()
            images_rgb.append(''.join(path +
                                      image_name +
                                      '.jpg'))
            images_depth.append(''.join(path +
                                        image_name.replace('rgb/jpg',
                                                           'depth') +
                                        '.jpg'))
            images_label_axis.append((float(pos_x),
                                      float(pos_y),
                                      # float(pos_z)
                                      ))
            images_label_quat.append((float(quat_w),
                                      # float(quat_p),
                                      # float(quat_q),
                                      float(quat_r)
                                      ))

    return images_rgb, \
        images_depth, \
        images_label_axis, \
        images_label_quat

# '../dataset/laser/route1/' + 'rgb/jpg/' +  + 011027 + jpg
# <      command line      >
