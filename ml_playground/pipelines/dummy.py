from __future__ import absolute_import

from ml_playground.data.dummy import DummyData
from ml_playground.models.tensorflow.tf_softmax import TfSoftmax
from ml_playground.utils.visual import PrintModels


def dummy_pipeline():
    data = DummyData(dim=10, training_num=20000, test_num=2000, noise=0.1)
    data.Init()
    model = TfSoftmax(data, silent=True)
    model.Train()
    model.Eval()
    PrintModels([model])
