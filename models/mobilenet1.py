import os
import math
import numpy as np
import tensorflow as tf

from keras.applications import mobilenet
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras import optimizers
from keras.preprocessing import image
from keras.utils import to_categorical

import tensorflow_datasets as tfds

def mobilenet1():
    # Load MobileNet model
    model = MobileNet(weights='imagenet')
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

