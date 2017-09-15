from __future__ import absolute_import

import time
import tensorflow as tf

from ml_playground.models.base import Model
from ml_playground.utils.visual import PrintProgress
from ml_playground.models.tensorflow.tensorflow_layer import (
    ConvLayer, DenselyConnectedLayer, DropoutLayer, ReadoutLayer, GetNetwork
)


class TfSimpleNetworkLayers(Model):
    def __init__(self, data, batch_size=100, steps=1000, **args):
        super(TfSimpleNetworkLayers, self).__init__(data, **args)
        self.name = 'TF Conv Network layer'
        self.batch_size = batch_size
        self.steps = steps
        self.sess = None
        self.W = None
        self.b = None

    def Train(self):
        self.metrics['accuracy'] = 'N/A'
        start_time = time.time()

        x = tf.placeholder(tf.float32, [None, self.data.dims])
        y_ = tf.placeholder(tf.float32, [None, self.data.max_label + 1])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        keep_prob = tf.placeholder(tf.float32)

        layers = [
            ConvLayer(
                patch_height=5,
                patch_width=5,
                input_features_num=1,
                output_features_num=32,
                max_pool_args={
                    "ksize": [1, 2, 2, 1],
                    "strides": [1, 2, 2, 1], 
                    "padding": 'SAME',
                },
            ),
            ConvLayer(
                patch_height=5,
                patch_width=5,
                input_features_num=32,
                output_features_num=64,
                max_pool_args={
                    "ksize": [1, 2, 2, 1],
                    "strides": [1, 2, 2, 1], 
                    "padding": 'SAME',
                },
            ),
            DenselyConnectedLayer(
                input_neuron_num=7 * 7 * 64,
                output_neuron_num=1024,
            ),
            DropoutLayer(
                keep_prob=keep_prob,
            ),
            ReadoutLayer(
                input_neuron_num=1024,
                output_neuron_num=10,
            ),
        ]

        y_conv = GetNetwork(x_image, layers)

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
        if self.sess is None:
            raise Exception("Model is not trained yet.")
        pass

    def GetDebugTable(self):
        headers, data = super(TfSimpleNetworkLayers, self).GetDebugTable()
        addtional_headers = ["batch_size", "steps"]
        additonal_data = [self.batch_size, self.steps]
        return headers + addtional_headers, data + additonal_data
