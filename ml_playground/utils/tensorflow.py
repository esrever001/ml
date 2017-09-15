
from __future__ import absolute_import

import tensorflow as tf

def MaxPool(x, ksize, strides):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')