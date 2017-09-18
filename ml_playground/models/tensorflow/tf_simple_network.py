from __future__ import absolute_import

import time
import tensorflow as tf

from ml_playground.models.base import Model
from ml_playground.utils.visual import PrintProgress


class TfSimpleNetwork(Model):
    def __init__(self, data, batch_size=100, steps=1000, **args):
        super(TfSimpleNetwork, self).__init__(data, **args)
        self.name = 'TF Conv Network'
        self.batch_size = batch_size
        self.steps = steps
        self.sess = None
        self.W = None
        self.b = None

    def Train(self):
        self.metrics['accuracy'] = 'N/A'
        start_time = time.time()

        x = tf.placeholder(tf.float32, [None, self.data.dims])
        self.W = tf.Variable(tf.zeros([self.data.dims, self.data.max_label + 1]))
        self.b = tf.Variable(tf.zeros([self.data.max_label + 1]))
        y_ = tf.placeholder(tf.float32, [None, self.data.max_label + 1])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        for index in range(self.steps):
            if not self.silent:
                PrintProgress(index + 1, self.steps, "Training")
            batch_xs, _, batch_ys = self.data.GetNextBatch(self.batch_size)
            self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        elapsed_time = time.time() - start_time
        self.metrics['training_time'] = elapsed_time

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.metrics['accuracy'] = accuracy.eval(
            feed_dict={
                x: self.data.test_instances,
                y_: self.data.test_one_hot_labels,
                keep_prob: 1.0
            }
        )

    def Eval(self):
        if self.sess is None or self.W is None or self.b is None:
            raise Exception("Model is not trained yet.")
        pass

    def GetDebugTable(self):
        headers, data = super(TfSimpleNetwork, self).GetDebugTable()
        addtional_headers = ["batch_size", "steps"]
        additonal_data = [self.batch_size, self.steps]
        return headers + addtional_headers, data + additonal_data


def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
