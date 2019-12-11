import matplotlib.pyplot as plt
import config as cfg


def export_pose_loss_acc_curves(outcomes, axis_x, flag):
    cfg.log(export_pose_loss_acc_curves.__module__,
            export_pose_loss_acc_curves.__name__)
    plt.figure(figsize=(8, 5))
    plt.plot(axis_x, outcomes.history['cls3_fc_pose_xyz_' + flag])
    plt.plot(axis_x, outcomes.history['val_cls3_fc_pose_xyz_' + flag])
    plt.plot(axis_x, outcomes.history['cls3_fc_pose_wpqr_' + flag])
    plt.plot(axis_x, outcomes.history['val_cls3_fc_pose_wpqr_' + flag])
    plt.xlabel('epochs')
    plt.ylabel(flag)
    plt.title('epochs - ' + flag)
    plt.grid(True)
    plt.legend(['train', 'val'], loc=7)
    plt.style.use(['ggplot'])
    plt.savefig(cfg.OUTPUT_RESULT + 'acc_loss/pose_' + flag + '_curves.png')


def export_depth_loss_acc_curve(outcomes, axis_x, flag):
    cfg.log(export_depth_loss_acc_curve.__module__,
            export_depth_loss_acc_curve.__name__)
    plt.figure(figsize=(8, 5))
    plt.plot(axis_x, outcomes.history['cls_depth_' + flag])
    plt.plot(axis_x, outcomes.history['val_cls_depth_' + flag])
    plt.xlabel('epochs')
    plt.ylabel(flag)
    plt.title('epochs - ' + flag)
    plt.grid(True)
    plt.legend(['train', 'val'], loc=7)
    plt.style.use(['ggplot'])
    plt.savefig(cfg.OUTPUT_RESULT + 'acc_loss/depth_' + flag + '_curves.png')


def export_curves_and_depth(outcomes, epochs):
    axis_x = range(epochs + 1)
    export_pose_loss_acc_curves(outcomes, axis_x, 'loss')
    export_pose_loss_acc_curves(outcomes, axis_x, 'accuracy')
    export_depth_loss_acc_curve(outcomes, axis_x, 'loss')
    export_depth_loss_acc_curve(outcomes, axis_x, 'accuracy')


def export_depth_comparison(y_network, y_test, num=3):
    cfg.log(export_depth_comparison.__module__,
            export_depth_comparison.__name__)
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

    plt.savefig(cfg.OUTPUT_RESULT + 'depth/depth_comparison.png')
