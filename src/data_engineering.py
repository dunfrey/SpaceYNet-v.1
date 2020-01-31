import numpy
import cv2
from sklearn.preprocessing import StandardScaler

import utils
import config


def extract_data(method, path, mode=None):
    utils.save_log(f'{extract_data.__module__} :: '
                   f'{extract_data.__name__}')

    print('\nExtracting data from text file')

    images_rgb, images_depth, labels_axis, labels_quat = \
        extract_data_from_file(method, path, mode=mode)

    print('Extracting RGB images')
    images_rgb = Processer.extract_images(images_rgb)

    if mode == 'online':
        return images_rgb, None, None, None

    print('Extracting Depth-scene images')
    images_depth = Processer.extract_images(images_depth)

    labels_axis, labels_quat = \
        Processer.pose_standardization(labels_axis, labels_quat)

    return images_rgb, \
        images_depth, \
        labels_axis, \
        labels_quat


def extract_data_from_file(method, path, mode=None):
    utils.save_log(f'{extract_data_from_file.__module__} :: '
                   f'{extract_data_from_file.__name__}')

    images_rgb = []
    images_depth = []
    images_label_axis = []
    images_label_quat = []

    file = 'dataset_train.txt' \
        if (method == 'train') \
        else 'dataset_test.txt'

    with open(path + file) as f:
        for line in f:
            (image_name, pos_x, pos_y, pos_z,
             quat_w, quat_p, quat_q, quat_r) = line.split()
            images_rgb.append(''.join(path +
                                      image_name +
                                      '.jpg'))
            images_depth.append(''.join(path +
                                        image_name.replace('rgb',
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

    if mode is 'online':
        _, last = utils.sizeof_train_validation_data(len(images_rgb))
        return images_rgb[:last], None, None, None

    return images_rgb, \
        images_depth, \
        images_label_axis, \
        images_label_quat


class Processer:

    @staticmethod
    def extract_images(input_images):
        utils.save_log(f'{Processer.extract_images.__module__} :: '
                       f'{Processer.extract_images.__name__}')
        images = numpy.zeros((len(input_images),
                              config.IMG_HEIGHT,
                              config.IMG_WIDTH,
                              config.IMG_CHANNEL),
                             dtype=numpy.uint8)
        for i in range(len(input_images)):
            image = cv2.imread(input_images[i])
            image = cv2.resize(image, (config.IMG_HEIGHT,
                                       config.IMG_WIDTH))
            images[i] = image
        images = images.astype('float32')
        images /= 255
        return images

    @staticmethod
    def pose_standardization(input_axis, input_quat):
        utils.save_log(
            f'{Processer.pose_standardization.__module__} :: '
            f'{Processer.pose_standardization.__name__}')

        input_axis = numpy.asarray(input_axis)
        scaler_axis = StandardScaler().fit(input_axis)
        rescaled_axis = scaler_axis.transform(input_axis)

        out_axis = numpy.zeros((len(input_axis), 2),
                               dtype=numpy.uint8)
        out_axis = rescaled_axis[:, 0:2]

        input_quat = numpy.asarray(input_quat)
        scaler_quat = StandardScaler().fit(input_quat)
        rescaled_quat = scaler_quat.transform(input_quat)

        out_quat = numpy.zeros((len(input_quat), 2),
                               dtype=numpy.uint8)
        out_quat = rescaled_quat[:, 0:2]

        return out_axis, out_quat
