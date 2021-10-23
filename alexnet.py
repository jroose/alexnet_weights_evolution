# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings
#



# Press the green button in the gutter to run the script.

# import matplotlib as plt
# import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity

def create_alexnet():
    # input layer: size (224, 224, 3)
    X = input(X_input)

    # Five convolutional layers and  #two Max pooling layer, parallel:.  Followed by Relu? padding???

    # First CL: 96 kernels of size (11, 11, 3), stride 4
    X = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same', kernel_initializer=random_uniform())(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Second CL: 256 kernels of size (5, 5, 48), stride
    X = Conv2D(filters=256, kernel_size=(5, 5), strides=(4, 4), padding='same', kernel_initializer=random_uniform())(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Third CL: 384 kernels of size (3, 3, 256), stride
    X = Conv2D(filters=384, kernel_size=(3, 3), strides=(4, 4), padding='same', kernel_initializer=random_uniform())(X)
    X = Activation('relu')(X)

    # Forth CL: 384 kernels of size (3, 3, 192), stride
    X = Conv2D(filters=384, kernel_size=(3, 3), strides=(4, 4), padding='same', kernel_initializer=random_uniform())(X)
    X = Activation('relu')(X)

    # Fifth CL: 256 kernels of size (3, 3, 192), stride
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(4, 4), padding='same', kernel_initializer=random_uniform())(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Three fully connected layers with 4096 neurons
    X = Dense(4096, activation='relu', kernel_initializer=glorot_uniform())(X)
    X = Dense(4096, activation='relu', kernel_initializer=glorot_uniform())(X)
    X = Dense(1000, activation='softmax', kernel_initializer=glorot_uniform())(X)


if __name__ == '__main__':
    #import the photos from external library
    #X_input

    #preprocess the photos and data augmentation


    #define layers of network

    X = create_alexnet()
    model = Model(inputs=X_input, outputs=X)

    #Train model, classify images

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
