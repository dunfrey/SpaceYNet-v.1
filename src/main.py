import sys

import system_configurator
import data_engineering
import exporter
import networks
import config
import utils


def train(*args, **kwargs):
    """Execute:
    $ python main.py \
    train \
    --path_data_train ../dataset/laser/ \
    --output_path ../output/
    """
    utils.save_log(f'{train.__module__} :: {train.__name__}')

    model = system_configurator.set_network(kwargs['output_path'])
    checkpointer = networks.set_checkpointer(kwargs['output_path'])
    lr_reducer = networks.set_reducer()

    _, X_img, y_depth, y_axis, y_quat = \
        data_engineering.extract_data(sys.argv[1],
                                      kwargs['path_data_train'])

    # fit nn spaceynet
    outcomes = model.fit(X_img,
                         [y_depth, y_axis, y_quat],
                         batch_size=config.BATCH_SIZE,
                         epochs=config.EPOCHS,
                         shuffle=True,
                         callbacks=[checkpointer, lr_reducer])

    exporter.export_curves_and_depth(outcomes,
                                     kwargs['output_path'])

    exporter.export_lr_curve(outcomes,
                             kwargs['output_path'],
                             config.EPOCHS)


def test(*args, **kwargs):
    """Execute:
    $ python main.py \
    test \
    --path_data_test ../dataset/laser/ \
    --output_path ../output/ \
    --mode offline
    """
    utils.save_log(f'{test.__module__} :: {test.__name__}')

    model = system_configurator.set_network(kwargs['output_path'])

    img_label, X_img, y_depth, y_axis, y_quat = \
        data_engineering.extract_data(sys.argv[1],
                                      kwargs['path_data_test'])

    outcomes = model.predict(X_img)

    exporter.export_pose_prediction(img_label,
                        y_axis,
                        y_quat,
                        outcomes,
                        kwargs['output_path'])

    # print/save depth original and depth prediction
    exporter.export_depth_prediction(y_depth, outcomes, kwargs['output_path'])


def cli():
    """Call fire cli"""
    import fire
    return fire.Fire()


if __name__ == '__main__':
    cli()
