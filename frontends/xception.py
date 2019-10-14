import tensorflow as tf
import tensorflow.keras

from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization, Input, GlobalAveragePooling2D

def conv2d_block(x, filters, kernel_size, padding = 'same', strides = 1, activation = None):
    x = Conv2D(filters, kernel_size, padding = padding, strides = strides)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)

    return x

def separableconv2d_block(x, filters, kernel_size, padding = 'same', strides = 1, activation = None):
    x = SeparableConv2D(filters, kernel_size, padding = padding, strides = strides, depth_multiplier = 1)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)

    return x

def xception(inputs, num_classes, activation='softmax'):

    ######## entry flow ########
    x = conv2d_block(input, 32, 3, strides = 2, activation = 'relu')
    x = conv2d_block(x, 64, 3, activation = 'relu')
    filters = [128, 256, 728];
    
    for filter in filters:
        residual_network = conv2d_block(x, filter, 1, strides = 2)

        if filter != filters[0]:
            x = Activation(activation = 'relu')(x)

        x = separableconv2d_block(x, filter, 3, activation = 'relu')
        x = separableconv2d_block(x, filter, 3)

        x = MaxPooling2D(3, strides = 2, padding = 'same')(x)

        x = Add()([x, residual_network])

    ######## middle flow ########
    for i in range(8):
        x = Activation(activation = 'relu')(x)
        x = separableconv2d_block(x, 728, 3, activation = 'relu')
        x = separableconv2d_block(x, 728, 3, activation = 'relu')
        x = separableconv2d_block(x, 728, 3)

        x = Add()([x, residual_network])

    ######## exit flow ########
    residual_network = conv2d_block(x, filters, 1, strides = 2)

    x = Activation(activation = 'relu')(x)
    x = separableconv2d_block(x, 728, 3, activation = 'relu')
    x = separableconv2d_block(x, 1024, 3)
    x = MaxPooling2D(3, strides = 2, padding = 'same')(x)

    x = Add()([x, residual_network])

    x = separableconv2d_block(x, 1536, 3, activation = 'relu')
    x = separableconv2d_block(x, 2048, 3, activation = 'relu')

    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation = activation)(x)
    model = Model(inputs, output, name = 'Xception')

    return model
