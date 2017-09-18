from __future__ import absolute_import

import tensorflow as tf

from ml_playground.data.mnist import MNIST
from ml_playground.models.tensorflow.tf_softmax import TfSoftmax
from ml_playground.models.tensorflow.tf_simple_network import TfSimpleNetwork
from ml_playground.models.tensorflow.tf_simple_network_layers import TfSimpleNetworkLayers
from ml_playground.utils.visual import PrintModels
from ml_playground.models.tensorflow.tensorflow_layer import (
    ConvLayer, DenselyConnectedLayer, DropoutLayer, ReadoutLayer
)


def mnist_pipeline():
    data = MNIST()
    data.Init()
    keep_prob = tf.placeholder(tf.float32)
    models = [
        TfSoftmax(data, silent=True,  batch_size=50, steps=1000),
        TfSoftmax(data, silent=True,  batch_size=100, steps=100),
        TfSimpleNetworkLayers(
            data, silent=False, batch_size=50, steps=10000,
            input_reshape=[-1, 28, 28, 1],
            tf_extra_train_args={
                keep_prob: 0.5,
            },
            tf_extra_test_args={
                keep_prob: 1.0,
            },
            tf_extra_variables={
                'keep_prob': keep_prob,
            },
            layers=[
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
            ],
        ),
        TfSimpleNetwork(data, silent=False,  batch_size=50, steps=1000),
    ]
    for model in models:
        model.Train()
        model.Eval()
    PrintModels(models)
