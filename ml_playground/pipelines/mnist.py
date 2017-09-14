from __future__ import absolute_import

from ml_playground.data.mnist import MNIST
from ml_playground.models.tensorflow.tf_softmax import TfSoftmax
from ml_playground.models.tensorflow.tf_simple_network import TfSimpleNetwork
from ml_playground.utils.visual import PrintModels


def mnist_pipeline():
    data = MNIST()
    data.Init()
    models = [
        TfSoftmax(data, silent=True,  batch_size=50, steps=1000),
        TfSoftmax(data, silent=True,  batch_size=100, steps=100),
        TfSoftmax(data, silent=True,  batch_size=100, steps=1000),
        TfSoftmax(data, silent=True,  batch_size=100, steps=20000),
        # TfSimpleNetwork(data, silent=False,  batch_size=50, steps=20000),
        # TfSimpleNetwork(data, silent=False,  batch_size=100, steps=20000),
    ]
    for model in models:
        model.Train()
        model.Eval()
    PrintModels(models)
