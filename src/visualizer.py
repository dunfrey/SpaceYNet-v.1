import config_file
import utils

import matplotlib.pyplot as plt


def export_pose_loss_acc_curves(outcomes, path, epochs, flag):
    utils.save_log(f'{export_pose_loss_acc_curves.__module__} :: '
                   f'{export_pose_loss_acc_curves.__name__}')
    plt.figure(figsize=(8, 5))
    plt.xlabel('epochs')
    plt.ylabel(flag)
    plt.title('epochs - ' + flag)
    plt.grid(True)
    plt.legend(['train', 'val'], loc=7)
    plt.style.use(['ggplot'])

    plt.plot(epochs, outcomes.history['cls_fc_position_' + flag])
    plt.plot(epochs, outcomes.history['val_cls_fc_position_' + flag])
    plt.savefig(path + 'acc_loss/position_' + flag + '.png')

    plt.cla()

    plt.plot(epochs, outcomes.history['cls_fc_quaternion_' + flag])
    plt.plot(epochs, outcomes.history['val_cls_fc_quaternion_' + flag])
    plt.savefig(path + 'acc_loss/quaternion_' + flag + '.png')


def export_depth_loss_acc_curve(outcomes, path, epochs, flag):
    utils.save_log(f'{export_depth_loss_acc_curve.__module__} :: '
                   f'{export_depth_loss_acc_curve.__name__}')
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, outcomes.history['cls_depth_' + flag])
    plt.plot(epochs, outcomes.history['val_cls_depth_' + flag])
    plt.xlabel('epochs')
    plt.ylabel(flag)
    plt.title('epochs - ' + flag)
    plt.grid(True)
    plt.legend(['train', 'val'], loc=7)
    plt.style.use(['ggplot'])
    plt.savefig(path + 'acc_loss/depth_' + flag + '.png')


def export_lr_curve(outcomes, path, epochs):
    utils.save_log(f'{export_lr_curve.__module__} :: '
                   f'{export_lr_curve.__name__}')
    plt.figure(figsize=(8, 5))
    print(outcomes.history)
    plt.plot(epochs, outcomes.history['lr'])
    plt.xlabel('epochs')
    plt.grid(True)
    plt.legend(['Learning Rate over Time'], loc=7)
    plt.style.use(['ggplot'])
    plt.savefig(path + 'acc_loss/learning_rate.png')


def export_curves_and_depth(outcomes, path):
    epochs = range(config_file.EPOCHS)

    export_lr_curve(outcomes, path, epochs)

    export_pose_loss_acc_curves(outcomes,
                                path,
                                epochs,
                                'loss')

    export_pose_loss_acc_curves(outcomes,
                                path,
                                epochs,
                                'accuracy')

    export_depth_loss_acc_curve(outcomes,
                                path,
                                epochs,
                                'loss')

    export_depth_loss_acc_curve(outcomes,
                                path,
                                epochs,
                                'accuracy')


def export_depth_comparison(y_network, y_test, num=3):
    utils.save_log(f'{export_depth_comparison.__module__} :: '
                   f'{export_depth_comparison.__name__}')
    plt.figure(figsize=(12, 8))

    for i in range(num):
        # y_distances : distance between figures that is printed
        # first pair: 1-y_test and 1-y_network
        # second pair: 100-y_test and 100-y_network
        y_distances = i * 100

        # display original
        ax = plt.subplot(2, num, i + 1)
        plt.imshow(y_network[y_distances].reshape(cfg.IMG_WIDTH, cfg.IMG_HEIGHT, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, num, i + 1 + num)
        plt.imshow(y_test[0][y_distances].reshape(cfg.IMG_WIDTH, cfg.IMG_HEIGHT, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(config.OUTPUT_RESULT + 'depth/depth_comparison.png')
