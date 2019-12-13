from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv1D, BatchNormalization, Activation, add, Dense, Flatten


class ResnetBlock(Model):
    def __init__(self, n_filters, n_kernels, is_training=False):
        super(ResnetBlock, self).__init__()

        self.is_training = is_training
        self.n_filters = n_filters
        self.n_kernels = n_kernels

        self.conv1 = Conv1D(self.n_filters, self.n_kernels[0], activation=None, padding='same', name='conv1')
        self.bn1 = BatchNormalization(name='batchnorm1')
        self.relu1 = Activation(activation='relu', name='relu1')

        self.conv2 = Conv1D(self.n_filters, self.n_kernels[1], activation=None, padding='same', name='conv2')
        self.bn2 = BatchNormalization(name='batchnorm2')
        self.relu2 = Activation(activation='relu', name='relu2')

        self.conv3 = Conv1D(self.n_filters, self.n_kernels[2], activation=None, padding='same', name='conv3')
        self.bn3 = BatchNormalization(name='batchnorm3')

        self.shortcut = Conv1D(self.n_filters, 1, activation=None, padding='same', name='shortcut')
        self.bn_shortcut = BatchNormalization(name='batchnorm_shortcut')
        self.out_block = Activation(activation='relu', name='out_block')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)
        x = add([x, shortcut])
        out_block = self.out_block(x)
        return out_block


class Resnet1D(Model):
    def __init__(self, params=None, is_training=False):
        super(Resnet1D, self).__init__()

        self.is_training = is_training
        self.n_cnn_filters = params['n_cnn_filters']
        self.n_cnn_kernels = params['n_cnn_kernels']
        self.n_fc_units = params['n_fc_units']
        self.n_classes = params['n_classes']

        # Resnet Blocks
        self.block1 = ResnetBlock(self.n_cnn_filters[0], self.n_cnn_kernels, is_training)
        self.block2 = ResnetBlock(self.n_cnn_filters[1], self.n_cnn_kernels, is_training)
        self.block3 = ResnetBlock(self.n_cnn_filters[2], self.n_cnn_kernels, is_training)
        self.block4 = ResnetBlock(self.n_cnn_filters[2], self.n_cnn_kernels, is_training)

        # Flatten
        self.flatten = Flatten(name='flatten')

        # FC
        self.fc1 = Dense(self.n_fc_units[0], activation='relu', name='fc1')
        self.fc2 = Dense(self.n_fc_units[1], activation='relu', name='fc2')
        self.fc3 = Dense(self.n_classes, activation=None, name='fc3')

    def call(self, inputs, training=None, mask=None):
        signal_input = inputs['features_input']

        with tf.name_scope('block1'):
            out_block1 = self.block1(signal_input)

        with tf.name_scope('block2'):
            out_block2 = self.block2(out_block1)

        with tf.name_scope('block3'):
            out_block3 = self.block3(out_block2)

        with tf.name_scope('block4'):
            out_block4 = self.block4(out_block3)

        x = self.flatten(out_block4)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output
