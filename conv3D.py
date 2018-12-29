
import tensorflow as tf
import numpy as np

from Define import *

def Convolution_3D(inputs):
    model_name = '3d_'
    conv_index = 1
    pool_index = 1

    x = inputs

    for index in range(len(LAYERS)):
        for i in range(LAYERS[index]):
            x = tf.layers.conv3d(inputs = x, filters = FEATURES[index], kernel_size = [3, 3, 3], padding = 'same', activation = tf.nn.relu, name = model_name + 'conv' + str(conv_index)); conv_index += 1
        x = tf.layers.max_pooling3d(inputs = x, pool_size = [2,2,2], strides = 2, name = model_name + 'pool' + str(pool_index)); pool_index += 1
        print(x)

    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(inputs = x, units = CLASSES, name = 'fc')

    return x

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, [None, SEQUENCE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'inputs')
    Y = tf.placeholder(tf.float32, [None, CLASSES], name = 'labels')

    pred = Convolution_3D(X)
    print(pred)