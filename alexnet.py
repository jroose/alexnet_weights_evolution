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
from  tensorflow.keras.optimizers import Adagrad, Adam
#from tensorflow.keras.
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import he_normal, glorot_uniform, constant, identity
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def create_alexnet():
    # input layer: size (224, 224, 3)
    X = Input(shape=(227, 227, 3))
    
    inputs = X

    #data augmentation, random flips and random rotation
    X = RandomFlip()(X)
    X = RandomRotation([0,1])(X)
    X = RandomZoom((-0.1, 0.1))(X)

    # Five convolutional layers and  #two Max pooling layer, parallel:.  Followed by Relu? padding???

    # First CL: 96 kernels of size (11, 11, 3), stride 4
    #X = (X - 116) / 70
    #X = BatchNormalization()(X)
    X = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', kernel_initializer=he_normal(),bias_initializer='ones')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Second CL: 256 kernels of size (5, 5, 48), stride
    #X = BatchNormalization()(X)
    X = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=he_normal(), bias_initializer='ones')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Third CL: 384 kernels of size (3, 3, 256), stride
    #X = BatchNormalization()(X)
    X = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(), bias_initializer='ones')(X)
    X = Activation('relu')(X)

    # Forth CL: 384 kernels of size (3, 3, 192), stride
    #X = BatchNormalization()(X)
    X = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(), bias_initializer='ones')(X)
    X = Activation('relu')(X)

    # Fifth CL: 256 kernels of size (3, 3, 192), stride
    #X = BatchNormalization()(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(), bias_initializer='ones')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = Flatten()(X)

    # Three fully connected layers, first two with 4096 neurons and Dropout
    #X = BatchNormalization()(X)
    X = Dense(4096, activation='relu', kernel_initializer=he_normal(), bias_initializer='ones')(X)
    X = Dropout(0.5)(X)
    X = Dense(4096, activation='relu', kernel_initializer=he_normal(), bias_initializer='ones')(X)
    X = Dropout(0.5)(X)
    X = Dense(1000, activation='softmax', kernel_initializer=he_normal())(X)


    model = Model(inputs=inputs, outputs = X, name= "alexnet")
    model.compile(optimizer = Adam(learning_rate=1e-4), loss = SparseCategoricalCrossentropy(), metrics = ['accuracy', 'mse'])

    return model



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
