import argparse
import numpy as np
import os
import time

from src.config import Configuration
from src.mnist_dataset import MnistDataset
from src.number_classifier import NumberClassifier


def parse_arguments():
    """ Parses all the argument of the command line.

    :return: the args
    """
    argparser = argparse.ArgumentParser(
        description="MNIST number classifier arguments.")

    argparser.add_argument(
        '-i',
        '--data_path',
        help="path of the data folder.",
        default="data")

    argparser.add_argument(
        '-n',
        '--normalize',
        type=str2bool,
        nargs='?',
        help="whether you want to normalize the dataset",
        default=True)

    argparser.add_argument(
        '-s',
        '--shuffle',
        type=str2bool,
        nargs='?',
        help="whether you want to shuffle the dataset",
        default=True)

    argparser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help="number of channels in the image representation",
        default=32)

    argparser.add_argument(
        '-c',
        '--nb_classes',
        type=int,
        help="number output classes.",
        default=10)

    argparser.add_argument(
        '-e',
        '--nb_epoch',
        type=int,
        help="number of epoch during the training",
        default=20)

    argparser.add_argument(
        '-vs',
        '--validation_split',
        type=float,
        help="validation split between 0 and 1",
        default=0.2)

    return argparser.parse_args()


def str2bool(v):
    """ Transform a string to a bool. It will be true if the string is "yes" or "true" or "t" or "y" or "1". Otherwise,
    it will be false.

    :param v: input string
    :return: result as a bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False


def script_mnist_number_classifier(arguments):
    """Script for training and evaluation the number classifier of the MNIST dataset

    :param arguments - arguments coming from the parser
    """
    # Create folders
    model_path = os.path.join(arguments.data_path, "models")
    os.makedirs(model_path, exist_ok=True)
    config_path = os.path.join(model_path, "configurations")
    os.makedirs(config_path, exist_ok=True)
    visualization_path = os.path.join(arguments.data_path, "visualization")
    os.makedirs(visualization_path, exist_ok=True)

    # Load dataset
    print("%%%%%%%% Loading dataset.")
    mnist_dataset = MnistDataset(normalize=True, nb_output_classes=10)
    x_train, y_train, x_test, y_test, image_height, image_width = mnist_dataset.get_dataset()

    labels_train = np.argmax(y_train, axis=1)
    MnistDataset.visualize(visualization_path, 10, 10, "train", x_train, labels_train)

    # Start the current model
    current_model_name = str(int(time.time()))
    current_model_path = os.path.join(model_path, current_model_name)
    os.makedirs(current_model_path, exist_ok=True)
    print("%%%%%%%% Starting model {}".format(current_model_name))

    # Create the config associated
    config = Configuration(
        current_model_name,
        image_height,
        image_width,
        arguments.normalize,
        arguments.shuffle,
        arguments.batch_size,
        arguments.nb_epoch,
        arguments.nb_classes,
        arguments.validation_split)
    config.save(config.get_filename(config_path, current_model_name))

    # Number classifier
    number_classifier = NumberClassifier()

    # Training
    number_classifier.train(x_train, y_train, config, current_model_path)

    # Evaluation
    number_classifier.evaluate(x_test, y_test, visualization_path)

    print("%%%%%%%% Done model {}".format(current_model_name))


if __name__ == "__main__":
    # Get the command line arguments
    arguments = parse_arguments()

    # Run the script to classify MNIST numbers
    script_mnist_number_classifier(arguments)


