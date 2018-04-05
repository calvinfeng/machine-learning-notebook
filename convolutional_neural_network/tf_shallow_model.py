import tensorflow as tf
import numpy as np


class ShallowModel(object):
    """A shallow model
    - 7x7 convolutional layer with 32 filters, stride = 1
    - ReLU
    - 7x7 convolutional layer with 32 filters, stride = 1
    - ReLU
    """
    def __init__(self, reg_strength=1e-3):
        tf.reset_default_graph()
        self.reg_strength = reg_strength

        # Define placeholders
        self.X = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)

        out_1 = tf.layers.conv2d(inputs=self.X,
                                 filters=32,
                                 kernel_size=[7, 7],
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
        out_2 = tf.layers.conv2d(inputs=out_1,
                                 filters=128,
                                 kernel_size=[7, 7],
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
        out_3 = tf.layers.max_pooling2d(inputs=out_2, pool_size=[2, 2], strides=2)
        out_4 = tf.layers.batch_normalization(inputs=out_3, axis=3, training=self.is_training)
        out_5 = tf.layers.dense(inputs=tf.reshape(out_4, [-1, 32*32*128]), units=1000, activation=tf.nn.relu)
        out_6 = tf.layers.dropout(inputs=out_5, rate=0.5, training=self.is_training)
        out_7 = tf.layers.dense(inputs=out_6, units=2)

        print out_1.shape
        print out_2.shape
        print out_3.shape
        print out_4.shape
        print out_5.shape
        print out_6.shape
        print out_7.shape

    def run(self, sess, inputX, inputY):
        print sess


if __name__ == '__main__':
    model = ShallowModel()
