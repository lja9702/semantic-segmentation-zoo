from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Dense, Activation, Input
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D

def conv2d_block(x, filters, kernel_size, padding='same', strides=(1, 1), activation=None):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)

    return x

def separableconv2d_block(x, filters, kernel_size, padding='same', strides=(1, 1), activation=None):
    x = SeparableConv2D(filters, kernel_size, padding=padding, strides=strides, use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)

    return x

def Xception(input=None, num_classes=1000, is_training=True, activation='softmax'):

    if input is None:
        input = Input(shape=(299, 299, 3))

    ######## entry flow ########
    x = conv2d_block(input, 32, (3, 3), strides=(2, 2), padding='valid', activation='relu')
    x = conv2d_block(x, 64, (3, 3), padding='valid', activation='relu')
    filters_list = [128, 256, 728]

    for filters in filters_list:
        residual_network = conv2d_block(x, filters, (1, 1), strides=(2, 2))

        if filters != filters_list[0]:
            x = Activation(activation='relu')(x)
        x = separableconv2d_block(x, filters, (3, 3), activation='relu')
        x = separableconv2d_block(x, filters, (3, 3))

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = tf.add(x, residual_network)

    ######## middle flow ########
    for i in range(8):
        residual_network = x

        x = Activation(activation='relu')(x)
        x = separableconv2d_block(x, 728, (3, 3), activation='relu')
        x = separableconv2d_block(x, 728, (3, 3), activation='relu')
        x = separableconv2d_block(x, 728, (3, 3))

        x = tf.add(x, residual_network)

    ######## exit flow ########
    residual_network = conv2d_block(x, 1024, (1, 1), strides=(2, 2))

    x = Activation(activation='relu')(x)
    x = separableconv2d_block(x, 728, (3, 3), activation='relu')
    x = separableconv2d_block(x, 1024, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = tf.add(x, residual_network)

    x = separableconv2d_block(x, 1536, (3, 3), activation='relu')
    x = separableconv2d_block(x, 2048, (3, 3), activation='relu')

    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation=activation)(x)

    model = models.Model(input, output, name='Xception')

    return model
