import config
import utils

import matplotlib.pyplot as plt


def export_pose_curves(outcomes, path, epochs, flag):
    utils.save_log(f'{export_pose_curves.__module__} :: '
                   f'{export_pose_curves.__name__}')
    plt.figure(figsize=(8, 5))
    plt.xlabel('epochs')
    plt.ylabel(flag)
    plt.grid(True)
    plt.legend(['train', 'val'], loc=7)

    plt.plot(epochs, outcomes.history['cls_position_' + flag])
    plt.plot(epochs, outcomes.history['val_cls_position_' + flag])
    plt.savefig(path + 'acc_loss/position_' + flag + '.png')

    plt.cla()

    plt.plot(epochs, outcomes.history['cls_quaternion_' + flag])
    plt.plot(epochs, outcomes.history['val_cls_quaternion_' + flag])
    plt.savefig(path + 'acc_loss/quaternion_' + flag + '.png')


def export_depth_curve(outcomes, path, epochs, flag):
    utils.save_log(f'{export_depth_curve.__module__} :: '
                   f'{export_depth_curve.__name__}')
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, outcomes.history['cls_depth_' + flag])
    plt.plot(epochs, outcomes.history['val_cls_depth_' + flag])
    plt.xlabel('epochs')
    plt.ylabel(flag)
    plt.grid(True)
    plt.legend(['train', 'val'], loc=7)
    plt.savefig(path + 'acc_loss/depth_' + flag + '.png')


def export_lr_curve(outcomes, path, epochs):
    utils.save_log(f'{export_lr_curve.__module__} :: '
                   f'{export_lr_curve.__name__}')
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, outcomes.history['lr'])
    plt.xlabel('epochs')
    plt.grid(True)
    plt.legend(['Learning Rate over Time'], loc=7)
    plt.savefig(path + 'acc_loss/learning_rate.png')


def export_depth_prediction(depth, regression, path, num=3):
    plt.figure(figsize=(12, 8))

    for i in range(num):
        # display original depth
        ax = plt.subplot(2, num, i + 1)
        plt.imshow(depth[i*500].reshape(config.IMG_WIDTH, 
                                        config.IMG_HEIGHT, 
                                        config.IMG_CHANNEL))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display model depth regression
        ax = plt.subplot(2, num, i + 1 + num)
        plt.imshow(regression[0][i*500].reshape(config.IMG_WIDTH, 
                                                config.IMG_HEIGHT, 
                                                config.IMG_CHANNEL))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)   
    
    plt.savefig(path + 'depth/depth_regression.png')


def export_pose_prediction(name_img, label_axis, label_quat, outcomes, path):

    f_results = open(path + 'pose/pose_outcomes.txt', "w+" )
    
    for i in range(len(label_axis)):
        angles = utils.quaternion_to_euler(outcomes[2][i][0],
                                     outcomes[2][i][1],
                                     outcomes[2][i][2],
                                     outcomes[2][i][3])

        axis = str(outcomes[1][i]) + "\n"

        quat = str(outcomes[2][i]) + "\n"

        angl = str(angles[0]) + " - " + 
               str(angles[1]) + " - " + 
               str(angles[2])
    
        f_results.write(str(name_img[i]) + "\n")
        f_results.write("Axis: " + axis)
        f_results.write("Quat: " + quat)
        f_results.write("Angles (yaw, pitch, roll): " + angl)

        f_results.write("\n--\n")
        
    f_results.close()


def export_curves_and_depth(outcomes, path):
    epochs = range(config.EPOCHS)

    export_lr_curve(outcomes, path, epochs)

    export_pose_curves(outcomes, path, epochs, 'loss')
    export_pose_curves(outcomes, path, epochs, 'accuracy')

    export_depth_curve(outcomes, path, epochs, 'loss')
    export_depth_curve(outcomes, path, epochs, 'accuracy')
