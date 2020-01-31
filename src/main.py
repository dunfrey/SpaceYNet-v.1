import sys

import system_configurator
import data_engineering
import networks
import utils


def train(*args, **kwargs):
    """Execute:
    $ python main.py \
    train \
    --input_data_path ../dataset/laser/ \
    --output_path ../output/
    """
    utils.save_log(f'{train.__module__} :: {train.__name__}')

    model = system_configurator.set_network(kwargs['output_path'])
    checkpointer = networks.set_checkpointer(kwargs['output_path'])
    lr_reducer = networks.set_reducer()

    X_img, y_depth, y_axis, y_quat = \
        data_engineering.extract_data(sys.argv[1],
                                      kwargs['input_data_path'])


def test(*args, **kwargs):
    """Execute:
    $ python main.py \
    test \
    --input_data_path ../dataset/laser/ \
    --output_path ../output/ \
    --mode offline
    """
    utils.save_log(f'{test.__module__} :: {test.__name__}')


def cli():
    """Call fire cli"""
    import fire
    return fire.Fire()


if __name__ == '__main__':
    cli()