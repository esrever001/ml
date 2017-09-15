from __future__ import absolute_import

import tensorflow as tf


class Layer(object):
    def __init__(self, input_layer=None, pool_func=None, acitivate_func=None):
        self.input_layer = input_layer
        self.pool_func = pool_func
        self.acitivate_func = acitivate_func
        self.output_layer = None

    def GetOutputLayer(self):
        self.output_layer = self.output_layer if self.output_layer is not None else self.input_layer
        self.output_layer = self.acitivate_func(self.output_layer) if self.acitivate_func is not None else self.output_layer
        self.output_layer = self.pool_func(self.output_layer) if self.pool_func is not None else self.output_layer
        return self.output_layer


class ConvLayer(Layer):
    def __init__(
        self, patch_height, patch_width,
        input_features_num, output_features_num,
        conv_args={'strides': [1, 1, 1, 1], 'padding': 'SAME'},        
        acitivate_func=tf.nn.relu,
        max_pool_args=None, **args
    ):
        pool_func = (lambda x: tf.nn.max_pool(x, **max_pool_args)) if max_pool_args is not None else None
        super(ConvLayer, self).__init__(acitivate_func=acitivate_func, pool_func=pool_func, **args)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.input_features_num = input_features_num
        self.output_features_num = output_features_num
        self.weight = None
        self.bias = None
        self.conv_args = conv_args

    def GetOutputLayer(self):
        conv_layer = self.input_layer
        shape = [self.patch_height, self.patch_width, self.input_features_num, self.output_features_num]
        self.weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.output_features_num]))
        conv_layer = tf.nn.conv2d(self.input_layer, self.weight, **self.conv_args)

        self.output_layer = conv_layer + self.bias
        return super(ConvLayer, self).GetOutputLayer()


class DenselyConnectedLayer(Layer):
    def __init__(self, input_neuron_num, output_neuron_num, **args):
        super(DenselyConnectedLayer, self).__init__(**args)
        self.input_neuron_num = input_neuron_num
        self.output_neuron_num = output_neuron_num
        self.weight = None
        self.bias = None

    def GetOutputLayer(self):
        self.weight = tf.Variable(tf.truncated_normal([self.input_neuron_num, self.output_neuron_num], stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.output_neuron_num]))
        input_flat_layer = tf.reshape(self.input_layer, [-1, self.input_neuron_num])

        self.output_layer = tf.matmul(input_flat_layer, self.weight) + self.bias
        return super(DenselyConnectedLayer, self).GetOutputLayer()


class DropoutLayer(Layer):
    def __init__(self, keep_prob, **args):
        super(DropoutLayer, self).__init__(**args)
        self.keep_prob = keep_prob

    def GetOutputLayer(self):
        self.output_layer = tf.nn.dropout(self.input_layer, self.keep_prob)
        return super(DropoutLayer, self).GetOutputLayer()


class ReadoutLayer(Layer):
    def __init__(self, input_neuron_num, output_neuron_num, **args):
        super(ReadoutLayer, self).__init__(**args)
        self.input_neuron_num = input_neuron_num
        self.output_neuron_num = output_neuron_num
        self.weight = None
        self.bias = None

    def GetOutputLayer(self):
        self.weight = tf.Variable(tf.truncated_normal([self.input_neuron_num, self.output_neuron_num], stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.output_neuron_num]))

        self.output_layer = tf.matmul(self.input_layer, self.weight) + self.bias
        return super(ReadoutLayer, self).GetOutputLayer()


def GetNetwork(input_layer, layers):
    if not len(layers):
        return None
    previous_layer_output = input_layer
    for layer in layers:
        layer.input_layer = previous_layer_output
        previous_layer_output = layer.GetOutputLayer()
    return previous_layer_output


def __getMaxPoolFunc(**args):
    return lambda x: tf.nn.max_pool(x, **args)
