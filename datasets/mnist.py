from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import tensorflow as tf

class MNISTDataset:
    def __init__(self):
        self.name = "MNIST"
        self.image_size = 28
        self.num_channels = 1
        self.num_classes = 10

    def get_test_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        Y_test = np_utils.to_categorical(y_test, self.num_classes)
        del X_train, y_train
        return X_test, Y_test

    def get_val_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        val_size = 5000
        X_val = X_train[:val_size]
        X_val = X_val.reshape(X_val.shape[0], self.image_size, self.image_size, self.num_channels)
        X_val = X_val.astype('float32') / 255
        y_val = y_train[:val_size]
        Y_val = np_utils.to_categorical(y_val, self.num_classes)
        del X_train, y_train, X_test, y_test

        return X_val, Y_val

    def get_train_dataset(self):
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_train = np.repeat(x_train, 3, axis=-1)
        x_train = x_train.astype('float32') / 255
        x_train = tf.image.resize(x_train, [32,32]) # if we want to resize 
        y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)

        return x_train, y_train