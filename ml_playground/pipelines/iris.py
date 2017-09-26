from __future__ import absolute_import

from ml_playground.data.iris import Iris
from ml_playground.models.tensorflow.tf_softmax import TfSoftmax
from ml_playground.utils.visual import PrintModels


def iris_pipeline():
    data1 = Iris(ratio=0.2)
    data1.Init()
    data2 = Iris(ratio=0.5)
    data2.Init()
    models = [
        TfSoftmax(data1, silent=True, batch_size=50, steps=100),
        TfSoftmax(data1, silent=True, batch_size=150, steps=100),
        TfSoftmax(data1, silent=True, batch_size=150, steps=1000),
        TfSoftmax(data2, silent=True, batch_size=50, steps=100),
        TfSoftmax(data2, silent=True, batch_size=150, steps=100),
        TfSoftmax(data2, silent=True, batch_size=150, steps=1000),
    ]
    for model in models:
        model.Train()
        model.Eval()
    PrintModels(models)
