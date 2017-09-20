from __future__ import absolute_import

import time
import tensorflow as tf

from ml_playground.models.base import Model
from ml_playground.utils.visual import PrintProgress
from ml_playground.models.tensorflow.tensorflow_layer import GetNetwork


class TfSimpleNetworkLayers(Model):
    def __init__(self,
                 data,
                 layers,
                 input_reshape=None,
                 tf_extra_train_args={},
                 tf_extra_test_args={},
                 tf_extra_variables={},
                 batch_size=100,
                 steps=1000,
                 **args):
        super(TfSimpleNetworkLayers, self).__init__(data, **args)
        self.name = 'TF Conv Network layer'
        self.batch_size = batch_size
        self.steps = steps
        self.sess = None
        self.input_reshape = input_reshape
        self.tf_extra_train_args = tf_extra_train_args
        self.tf_extra_test_args = tf_extra_test_args
        self.tf_extra_variables = tf_extra_variables
        self.layers = layers

    def Train(self):
        self.metrics['accuracy'] = 'N/A'
        start_time = time.time()

        x = tf.placeholder(tf.float32, [None, self.data.dims])
        y_ = tf.placeholder(tf.float32, [None, self.data.max_label + 1])
        if self.input_reshape:
            x_input = tf.reshape(x, self.input_reshape)
        else:
            x_input = x

        y_conv = GetNetwork(x_input, self.layers)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        for index in range(self.steps):
            if not self.silent:
                PrintProgress(index + 1, self.steps, "Training")
            batch_xs, _, batch_ys = self.data.GetNextBatch(self.batch_size)
            feed_dict = {x: batch_xs, y_: batch_ys}
            feed_dict.update(self.tf_extra_train_args)
            self.sess.run(train_step, feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        self.metrics['training_time'] = elapsed_time

    def Eval(self):
        if self.sess is None:
            raise Exception("Model is not trained yet.")

        x = tf.placeholder(tf.float32, [None, self.data.dims])
        y_ = tf.placeholder(tf.float32, [None, self.data.max_label + 1])
        if self.input_reshape:
            x_input = tf.reshape(x, self.input_reshape)
        else:
            x_input = x

        y_conv = GetNetwork(x_input, self.layers)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        feed_dict = {
            x: self.data.test_instances,
            y_: self.data.test_one_hot_labels
        }
        feed_dict.update(self.tf_extra_test_args)
        self.metrics['accuracy'] = accuracy.eval(feed_dict=feed_dict, )

    def GetDebugTable(self):
        headers, data = super(TfSimpleNetworkLayers, self).GetDebugTable()
        addtional_headers = ["batch_size", "steps"]
        additonal_data = [self.batch_size, self.steps]
        return headers + addtional_headers, data + additonal_data
