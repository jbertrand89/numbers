from keras.datasets import mnist
from keras.utils import np_utils

import matplotlib.pyplot as plt
import numpy as np
import os


class MnistDataset(object):
    """ MNIST dataset """

    def __init__(self, normalize=True, nb_output_classes=10):
        """ Constructor

        :param normalize (bool) - if the dataset is normalized
        :param nb_output_classes (bool) - number of output classes. By default, it is 10 because there are 10 digits.
        """
        self._nb_output_classes = nb_output_classes
        self._normalized = False

        self._x_train, self._y_train, self._x_test, self._y_test = None, None, None, None
        self._nb_examples_train, self._nb_examples_test, self._image_height, self._image_width = -1, -1, -1, -1
        self._load_dataset()

        if normalize:
            self._normalized = normalize
            self._x_train = self.normalize(self._x_train)
            self._x_test = self.normalize(self._x_test)

    def _load_dataset(self):
        """"Loads the MNIST dataset in memory. Reshapes the x arrays and set them as floats. Transforms the y arrays as
        hot encoding.
        """
        # Load MNIST
        (self._x_train, self._y_train), (self._x_test, self._y_test) = mnist.load_data()

        # Transform to floats
        self._x_train = self._x_train.astype('float32')
        self._x_test = self._x_test.astype('float32')

        # Get shapes
        self._nb_examples_train = self._x_train.shape[0]
        self._nb_examples_test = self._x_test.shape[0]
        self._image_height = self._x_train.shape[1]
        self._image_width = self._x_train.shape[2]

        # Reshape
        self._x_train = self._x_train.reshape((self._nb_examples_train, self._image_height, self._image_width, 1))
        self._x_test = self._x_test.reshape((self._nb_examples_test, self._image_height, self._image_width, 1))

        # One hot encoding
        self._y_train = np_utils.to_categorical(self._y_train)
        self._y_test = np_utils.to_categorical(self._y_test)

    def get_dataset(self):
        """ Gets the dataset for training and testing

        :return: (np.array) - training images
        :return: (np.array) - training labels as hot encoding
        :return: (np.array) - test images
        :return: (np.array) - test labels as hot encoding
        :return: (int) - images height
        :return: (int) - images width
        """
        return self._x_train, self._y_train, self._x_test, self._y_test, self._image_height, self._image_width

    @staticmethod
    def normalize(x):
        """ Normalize the dataset. Images intensities are transformed from 0-255 to 0-1"""
        return x / 255

    @staticmethod
    def visualize(output_directory, n_rows, n_columns, dataset_name, images, labels, predictions=None):
        """ Visualizes the images with their labels and save the result into an image.

        :param output_directory (str): path of the directory where to save the file
        :param n_rows (int): number of images displayed per column
        :param n_columns (int): number of images displayed per row
        :param dataset_name (str): characteristic of the current images ("train", "test", "test_false" ...)
        :param images (np.array): images
        :param labels (np.array): ground truth labels (the labels are the digit value and not hot encoder)
        :param predictions (np.array): prediction labels (the labels are the digit value and not hot encoder)
        """
        plt.clf()
        total_elements = n_rows * n_columns

        image_height = images.shape[1]
        image_width = images.shape[2]

        for i in range(min(n_rows * n_columns, len(images))):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.subplots_adjust(hspace=.1, wspace=0.5)
            image_gray = images[i]
            image_gray = image_gray.reshape(image_height, image_width)
            image_rgb = np.stack((image_gray,) * 3, axis=-1)
            plt.imshow(image_rgb)

            label = labels[i]
            plt.text(1, 10, label, color=[0.3, 1, 0])  # print the ground truth label in green

            if predictions is not None:
                prediction = predictions[i]
                plt.text(20, 10, prediction, color="red")  # print the prediction label in blue

            plt.axis('off')

        plt.savefig(os.path.join(output_directory, '{0}_{1}.png'.format(dataset_name, total_elements)),
                    bbox_inches='tight')
