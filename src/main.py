import system_configurator
import utils
import config_file
import data_handler
import visualizer


def train(**kwargs):
    """Handle features and train model.

    Execute:
    $ python main.py \
    train \
    --input_data_path ../dataset/laser/ \
    --main_folder ../output/

    input_data_path: input dataset path
    output_path: output files path
    """
    utils.save_log(f'{train.__module__} :: {train.__name__}')

    model = system_configurator.set_network('train',
                                            kwargs['main_folder'])
    model_checkpoint = system_configurator.\
        create_checkpointer(kwargs['main_folder'])
    model_lr_reducer = system_configurator.create_reducer()

    X_img,  y_depth, y_axis, y_quat = \
        data_handler.extract_data('train',
                                  kwargs['input_data_path'])

    X_train, y_train_depth, y_train_axis, y_train_quat, \
        X_val, y_val_depth, y_val_axis, y_val_quat = \
        data_handler.split_train_val_data(X_img,
                                          y_depth,
                                          y_axis,
                                          y_quat)

    # fit model
    model_output = model.fit(X_train,
                             [y_train_depth,
                              y_train_axis,
                              y_train_quat],
                             validation_data=(X_val,
                                              [y_val_depth,
                                               y_val_axis,
                                               y_val_quat]),
                             batch_size=config_file.BATCH_SIZE,
                             epochs=config_file.EPOCHS,
                             shuffle=True,
                             callbacks=[model_checkpoint,
                                        model_lr_reducer])

    visualizer.export_curves_and_depth(model_output,
                                       kwargs['main_folder'])


def test(**kwargs):
    """Handle features and train model.

    Execute:
    $ python main.py \
    test \
    --mode offline
    --input_data_path ../dataset/laser/ \
    --main_folder ../output/

    mode: offline or online
    input_data_path: input dataset path
    output_path: output files path
    """
    utils.save_log(f'{test.__module__} :: {test.__name__}')


def cli():
    """Call fire cli"""
    import fire
    return fire.Fire()


if __name__ == '__main__':
    cli()
