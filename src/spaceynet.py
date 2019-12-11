import sys
import handler as hdl


def train(**kwargs):
    """
    $ python spaceynet.py train \
    --input_dataset_folder ../../../spaceynet/dataset/laser/ \
    --input_file_data_train ../../../spaceynet/dataset/laser/testando.txt \
    --input_output_folder ../../../spaceynet/output_laser/ \
    --input_checkpoint_file '../../../spaceynet/output_laser/checkpoint/spaceynet.h5'
    """
    hdl.set_configurations(**kwargs)
    if hdl.extract_text_file('TRAIN', **kwargs):
        print('No Image/Label Data was Found')
        sys.exit()

    hdl.extract_data()
    hdl.fit_model(**kwargs)

    pass


def test(**kwargs):
    """
    $ python spaceynet.py test \
    --input_dataset_folder ../../../spaceynet/dataset/laser/ \
    --input_file_data_train ../../../spaceynet/dataset/laser/testando.txt \
    --input_output_folder ../../../spaceynet/output_laser/ \
    --input_checkpoint_file '../../../spaceynet/output_laser/checkpoint/spaceynet.h5'
    """
    hdl.set_configurations(**kwargs)
    if hdl.extract_text_file('TEST', **kwargs):
        print('No Image/Label Data was Found')
        sys.exit()

    pass


def run(**kwargs):
    """
    $ python run \
    --input_data_file_train ../data/dataset_train.txt \
    --input_data_file_test ../data/dataset_test.txt \
    --input_model_file ../output_laser/checkpoint/spaceynet.h5
    """
    train(**kwargs)
    test(**kwargs)


def cli():
    """Call fire cli"""
    import fire
    return fire.Fire()


if __name__ == '__main__':
    cli()
