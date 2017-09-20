from __future__ import absolute_import

import time
import tensorflow as tf

from ml_playground.models.base import Model
from ml_playground.utils.visual import PrintProgress


class TfSoftmax(Model):
    def __init__(self, data, batch_size=100, steps=1000, **args):
        super(TfSoftmax, self).__init__(data, **args)
        self.name = 'TF Softmax'
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
        y = tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        y_ = tf.placeholder(tf.float32, [None, self.data.max_label + 1])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        for index in range(self.steps):
            if not self.silent:
                PrintProgress(index + 1, self.steps, "Training")
            batch_xs, _, batch_ys = self.data.GetNextBatch(self.batch_size)
            self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        elapsed_time = time.time() - start_time
        self.metrics['training_time'] = elapsed_time

    def Eval(self):
        if self.sess is None or self.W is None or self.b is None:
            raise Exception("Model is not trained yet.")

        x = tf.placeholder(tf.float32, [None, self.data.dims])
        y = tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        y_ = tf.placeholder(tf.float32, [None, self.data.max_label + 1])

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy_eval = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = self.sess.run(accuracy_eval, feed_dict={x: self.data.test_instances, y_: self.data.test_one_hot_labels})
        self.metrics['accuracy'] = accuracy
        return accuracy

    def GetDebugTable(self):
        headers, data = super(TfSoftmax, self).GetDebugTable()
        addtional_headers = ["batch_size", "steps"]
        additonal_data = [self.batch_size, self.steps]
        return headers + addtional_headers, data + additonal_data
