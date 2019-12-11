import config as cfg
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def set_configurations(**kwargs):
    cfg.log(set_configurations.__module__,
            set_configurations.__name__)
    cfg.FOLDER_DATASET = kwargs['input_dataset_folder']
    cfg.OUTPUT_RESULT = kwargs['input_output_folder']


def get_category_exec(category, **kwargs):
    if category == 'TRAIN':
        return kwargs['input_file_data_train']
    elif category == 'TEST':
        return kwargs['input_file_data_test']
    else:
        cfg.log(get_category_exec.__module__,
                get_category_exec.__name__)
        import sys
        sys.exit()

def extract_text_file(category, **kwargs):
    cfg.log(extract_text_file.__module__,
            extract_text_file.__name__)

    input_file = get_category_exec(category, **kwargs)

    with open(input_file) as f:
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
            cfg.images_name = cfg.images_rgb
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
    cfg.images_label_axis, cfg.images_label_quat = \
        pose_standardization(cfg.images_label_axis, cfg.images_label_quat)
    cfg.images_rgb = extract_images(cfg.images_rgb)
    cfg.images_depth = extract_images(cfg.images_depth)


def pose_standardization(input_axis, input_quat):
    cfg.log(pose_standardization.__module__,
            pose_standardization.__name__)
    from sklearn.preprocessing import StandardScaler

    input_axis = np.asarray(input_axis)
    scaler_axis = StandardScaler().fit(input_axis)
    rescaled_axis = scaler_axis.transform(input_axis)

    input_quat = np.asarray(input_quat)
    scaler_quat = StandardScaler().fit(input_quat)
    rescaled_quat = scaler_quat.transform(input_quat)

    return rescaled_axis, rescaled_quat


def extract_images(input_images):
    cfg.log(extract_images.__module__,
            extract_images.__name__)
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


def fit_model(**kwargs):
    cfg.log(fit_model.__module__,
            fit_model.__name__)
    import visualizer as vls
    import models as mdl

    cfg.CHECKPOINTER = mdl.create_spaceynet(kwargs['input_checkpoint_file'])
    cfg.EARLIERSTOP = mdl.earlier_stop()
    model = mdl.create_spaceynet()

    outcomes = model.fit(cfg.images_rgb,
                         [cfg.images_depth,
                          cfg.images_label_axis,
                          cfg.images_label_quat],
                         validation_split=cfg.VALID_SPLIT,
                         batch_size=cfg.BATCH_SIZE,
                         epochs=cfg.EPOCHS,
                         shuffle=True,
                         callbacks=[cfg.CHECKPOINTER,
                                    cfg.EARLIERSTOP])

    model.save(cfg.CHECKPOINTER)
    vls.export_curves_and_depth(outcomes, cfg.EARLIERSTOP.stopped_epoch, **kwargs)


def export_pose_regression(, label_axis, label_quat, spaceynet, posenet):
    print(label_axis.shape)

    if spaceynet is not None:
        print("SHAPES SPACEYNET;\n")
        print(spaceynet[1].shape)
        print(spaceynet[1][0].shape)
        print(spaceynet[2].shape)
        print(spaceynet[2][0].shape)
        f_results = open(config.output_laser + 'pose/spaceynet_results_comparisons.txt', "w+")
    if spaceynet is None:
        print("SHAPES POSENET;\n")
        print(posenet[0].shape)
        print(posenet[1].shape)
        f_results = open(config.output_laser + 'pose/posenet_results_comparisons.txt', "w+")

    for i in range(len(label_axis)):
        angles = quaternion_to_euler(label_quat[i][0], 0, 0, label_quat[i][1])
        if spaceynet is not None:
            # angles_sn = quaternion_to_euler(spaceynet[2][i][0],spaceynet[2][i][1],spaceynet[2][i][2],spaceynet[2][
            # i][3])
            angles_sn = quaternion_to_euler(spaceynet[2][i][0], 0, 0, spaceynet[2][i][1])
            axis = "s: " + str(spaceynet[1][i]) + "\n"
            quat = "s: " + str(spaceynet[2][i]) + "\n"
            angl = "s: " + str(angles_sn[0]) + " - " + str(angles_sn[1]) + " - " + str(angles_sn[2])

            '''angles_sn = quaternion_to_euler(spaceynet[2][i][0],0,0,spaceynet[2][i][1])
            axis = str(spaceynet[1][i]) + "\n"
            quat = str(spaceynet[2][i]) + "\n"
            angl = str(angles_sn[0]) + " - " + str(angles_sn[1]) + " - " + str(angles_sn[2])'''
        if spaceynet is None:
            angles_pn = quaternion_to_euler(posenet[1][i][0], 0, 0,
                                            posenet[1][i][1])  # ,posenet[1][i][2],posenet[1][i][3])
            axis = str(posenet[0][i]) + "\n"
            quat = str(posenet[1][i]) + "\n"
            angl = str(angles_pn[0]) + " - " + str(angles_pn[1]) + " - " + str(angles_pn[2])

        f_results.write(str(name_img[i]) + "\n")
        f_results.write("Axis: " + str(label_axis[i]) + "\n")
        # f_results.write("Axis: ")
        f_results.write(axis)
        f_results.write("Quat: " + str(label_quat[i]) + "\n")
        # f_results.write("Quat: ")
        f_results.write(quat)
        f_results.write(
            "Angles - yaw: " + str(angles[0]) + " - pitch: " + str(angles[1]) + " - roll: " + str(angles[2]) + "\n")
        # f_results.write("Angles - yaw - pitch - roll\n")
        f_results.write(angl)

        f_results.write("\n--" + "\n")

    f_results.close()
    pass


def quaternion_to_euler(w, x, y, z):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]
