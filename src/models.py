from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential


def get_model(config):
    """ Creates the model from the configuration.

    :param config (Configuration) - configuration
    :return: (keras.engine.training.Model) - the model
    """
    model = get_model_layers(config)

    model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    return model


def get_model_layers(config):
    """ Creates the model layers from the configuration.

    :param config (Configuration) - configuration
    :return: (keras.engine.training.Model) - the model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(config.image_height, config.image_width, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(config.nb_classes, activation='softmax'))

    return model
