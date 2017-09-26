from __future__ import absolute_import

import tensorflow as tf

from ml_playground.data.cifar10 import CIFAR10
from ml_playground.models.tensorflow.tf_softmax import TfSoftmax
from ml_playground.models.tensorflow.tf_simple_network_layers import TfSimpleNetworkLayers
from ml_playground.utils.visual import PrintModels
from ml_playground.models.tensorflow.tensorflow_layer import (ConvLayer, DenselyConnectedLayer, DropoutLayer, ReadoutLayer)


def cifar10_pipeline():
    data = CIFAR10(ratio=0.5)
    data.Init()
    keep_prob = tf.placeholder(tf.float32)
    models = [
        TfSoftmax(data, silent=False, batch_size=500, steps=5000),
        TfSimpleNetworkLayers(
            data,
            silent=False,
            batch_size=500,
            steps=5000,
            input_reshape=[-1, 32, 32, 3],
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
                    input_features_num=3,
                    output_features_num=32,
                    max_pool_args={
                        "ksize": [1, 2, 2, 1],
                        "strides": [1, 2, 2, 1],
                        "padding": 'SAME',
                    }
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
                    }
                ),
                DenselyConnectedLayer(
                    input_neuron_num=8 * 8 * 64,
                    output_neuron_num=1024,
                ),
                DropoutLayer(keep_prob=keep_prob, ),
                ReadoutLayer(
                    input_neuron_num=1024,
                    output_neuron_num=10,
                ),
            ]
        )
    ]
    for model in models:
        model.Train()
        model.Eval()
    PrintModels(models)
