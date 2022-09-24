import numpy as np
import keras
import tensorflow as tf
import cv2 as cv
from keras.applications import mobilenet
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras import optimizers
from keras.preprocessing import image
from keras.utils import to_categorical

import tensorflow_datasets as tfds

class Imagenet2:
    def __init__(self):
        self.num_classes = 1000

    def get_test_dataset(self):
        imagenet = tfds.load('imagenet2012', split='validation', data_dir = '/Users/fhasan8/tensorflow_datasets/downloads/manual',
                                             download=False, shuffle_files=True,
                                             as_supervised=True, with_info=True)
        imagenet.download_and_prepare()
        # Download the data, prepare it, and write it to disk
        C = imagenet.info.features['label'].num_classes
        Nvalidation = imagenet.info.splits['validation'].num_examples
        Nbatch = 32
        assert C == 1000
        assert Nvalidation == 50000

        

        # Load data from disk as tf.data.Datasets
        validation_dataset =  datasets['validation']
        assert isinstance(validation_dataset, tf.data.Dataset)

        batch_size=32
        num_classes=1000

        images = np.zeros((batch_size, 224, 224, 3))
        labels = np.zeros((batch_size, num_classes))
        x_test, y_test = None, None
        for sample in tfds.as_numpy(validation_dataset[:5000]):
            image = sample["image"]
            label = sample["label"]

            images[count%batch_size] = mobilenet.preprocess_input(np.expand_dims(cv.resize(image, (224, 224)), 0))
            labels[count%batch_size] = np.expand_dims(to_categorical(label, num_classes=num_classes), 0)
            
            count += 1
            x_test, y_test = images, labels

        return x_test, y_test