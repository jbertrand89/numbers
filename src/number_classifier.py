from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support

from src.mnist_dataset import MnistDataset
from src.models import get_model


class NumberClassifier(object):

    def __init__(self):
        """Constructor
        """
        self._model = None
        self._model_history = None

    def train(self, x_train, y_train, config, model_path):
        """ Trains the number classifier.

        :param x_train (np.array) - images
        :param y_train (np.array) - labels as hot encodings
        :param config (Configuration) - configuration
        :param model_path (str) - path of the model folder where to save the models
        """
        # Get model
        self._model = get_model(config)

        # Checkpoint to save the models at each epoch
        checkpoint = ModelCheckpoint(
            os.path.join(model_path, "model_{epoch:05d}.h5"),
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            mode='min')

        best_checkpoint = ModelCheckpoint(
            os.path.join(model_path, "best_model.h5"),
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            mode='min')

        # Train the model
        self._model_history = self._model.fit(
            x_train, y_train,
            batch_size=config.batch_size,
            epochs=config.nb_epochs,
            validation_split=config.validation_split,
            shuffle=config.shuffle,
            callbacks=[checkpoint, best_checkpoint])

        # Plot history
        self._plot_history(model_path, config.model_name)

    def evaluate(self, x_test, y_test, output_path):
        """ Evaluates the model on the test set.

        :param x_test (np.array) - test images
        :param y_test (np.array) - test labels as hot encoding
        :param output_path (str) - path of the folder where to save the results.
        """
        if self._model is None:
            raise Exception("You must train the model before evaluating!")

        # Accuracy on the whole test set
        test_loss, test_accuracy = self._model.evaluate(x_test, y_test, verbose=0)
        print("Test score: loss={0} accuracy={1}".format(test_loss, test_accuracy))

        # Computes prediction and ground truth
        predictions = np.argmax(self._model.predict(x_test), axis=1)
        ground_truth = np.argmax(y_test, axis=1)

        # Accuracy, precision and recall by digit
        self._evaluate_by_digit(predictions, ground_truth)

        # Visualize the false predictions
        self._visualize_false_predictions(x_test, predictions, ground_truth, output_path)

    def _evaluate_by_digit(self, predictions, ground_truth):
        """ Compute the precision, recall, and F1 score for each digit.

        :param predictions (np.array) - test images as class value
        :param ground_truth (np.array) - test labels as class value
        """
        labels = range(np.max(ground_truth) + 1)

        precision, recall, f1_score, element_count = precision_recall_fscore_support(
            ground_truth, predictions, average=None, labels=labels)

        # Print the metrics
        for digit in labels:
            print("digit {0} precision {1} recall {2} f1 score {3} total elements {4}".format(
                digit, precision[digit], recall[digit], f1_score[digit], element_count[digit]))

    def _visualize_false_predictions(self, x_test, predictions, ground_truth, output_path):
        """ Visualize the false predictions and save them into a file.

        :param x_test (np.array) - images
        :param predictions (np.array) - test images as class value
        :param ground_truth (np.array) - test labels as class value
        :param output_path (str) - path of the folder where to save the result image.
        """
        indexes = np.where(predictions != ground_truth)
        print("False predictions {0} out of {1}".format(len(indexes[0]), x_test.shape[0]))

        dataset_name = "test_false"
        MnistDataset.visualize(
            output_path, 10, 10, dataset_name, x_test[indexes], ground_truth[indexes], predictions[indexes])

    def _plot_history(self, path, model_name):
        """Plots the history of the loss and accuracy and save them into files.

        :param path (str) - path of the folder where to save the history files
        :param model_name (str) - current model name
        """
        # plot loss
        self._plot_loss(path, model_name)

        # plot accuracy
        self._plot_accuracy(path, model_name)

    def _plot_loss(self, path, model_name):
        """Plots the history of the loss and save it into files.

        :param path (str) - path of the folder where to save the loss file
        :param model_name (str) - current model name
        """
        plt.clf()
        plt.title('Cross Entropy Loss')
        plt.plot(self._model_history.history['loss'], color='blue', label='train')
        plt.plot(self._model_history.history['val_loss'], color='orange', label='validation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(os.path.join(path, 'history_loss_{0}.png'.format(model_name)), bbox_inches='tight')

    def _plot_accuracy(self, path, model_name):
        """Plots the history of the accuracy and save it into files.

        :param path (str) - path of the folder where to save the accuracy file
        :param model_name (str) - current model name
        """
        plt.clf()
        plt.title('Accuracy')
        plt.plot(self._model_history.history['accuracy'], color='blue', label='train')
        plt.plot(self._model_history.history['val_accuracy'], color='orange', label='validation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(os.path.join(path, 'history_accuracy_{0}.png'.format(model_name)), bbox_inches='tight')
