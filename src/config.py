import json
import os


class Configuration(object):
    """ Configuration for one training of mnist number classification model.
    """
    def __init__(self, model_name, image_height, image_width, normalize, shuffle, batch_size, nb_epochs, nb_classes,
                 validation_split):
        """ Constructor

        :param model_name (str): name of the model
        :param image_height (int): height of the image
        :param image_width (int): width of the image
        :param normalize (bool): if the images are normalized in the generator
        :param shuffle (bool): if the generator shuffles the images
        :param batch_size (int): size of each batch of data
        :param nb_epochs (int): number of epochs during the training
        :param nb_classes (int): total number of classes in the model
        :param validation_split (float) : validation split (between 0 and 1)
        """
        self.model_name = model_name

        # data parameters
        self.image_height = image_height
        self.image_width = image_width
        self.normalize = normalize
        self.shuffle = shuffle

        # training parameters
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.validation_split = validation_split

        # model parameters
        self.nb_classes = nb_classes

    def save(self, filename):
        """Saves the current object into a file.

        :param filename (str) - full path of the configuration file
        """
        with open(filename, "w+") as writer:
            json.dump(self.__dict__, writer)

    @staticmethod
    def load(filename):
        """ Loads a configuration from a file.

        :param filename (str) - full path of the configuration file
        :return (SplineMRISliceModelConfiguration) - configuration
        """
        with open(filename, "r+") as reader:
            data = json.load(reader)

        model_name = data["model_name"]
        image_height = data["image_height"]
        image_width = data["image_width"]
        normalize = data["normalize"]
        shuffle = data["shuffle"]
        batch_size = data["batch_size"]
        nb_epochs = data["nb_epochs"]
        nb_classes = data["nb_classes"]
        validation_split = data["validation_split"]

        return Configuration(
            model_name, image_height, image_width, normalize, shuffle, batch_size, nb_epochs, nb_classes,
            validation_split)

    @staticmethod
    def get_filename(path, model_name):
        """ Gets the configuration file path associated to a given model name.

        :param path (str) - directory path
        :param model_name (str) - name of the model
        :return (str) - configuration file name associated to the model. It is the full path
        """
        return os.path.join(path, "config_{}.txt".format(model_name))
