import sys
sys.path.append(".")
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from datasets import * 
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import math

import os


def create_model_resnet50():
    input = Input(shape=(32,32,3))
    efnet = ResNet50(weights='imagenet', include_top = False, input_tensor = input)
    gap = GlobalMaxPooling2D()(efnet.output)
    output = Dense(10, activation='softmax', use_bias=True)(gap)
    model = Model(efnet.input, output)
    return efnet, model

def train():
    dataset = MNISTDataset()
    base_model, model = create_model_resnet50()
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    x_train, y_train = dataset.get_train_dataset()
    
    #model.compile(
    #      loss  = tf.keras.losses.CategoricalCrossentropy(),
    #      metrics = tf.keras.metrics.CategoricalAccuracy(),
    #      optimizer = tf.keras.optimizers.Adam())

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # fit 
    model.fit(x_train, y_train, verbose = 1, batch_size=10000, epochs=5, steps_per_epoch=6)

def resnet50_10():
    train()

if __name__ == '__main__':
    resnet50_10()