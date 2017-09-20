from __future__ import absolute_import

from ml_playground.data.base import Data
import numpy as np


class DummyData(Data):
    def __init__(self, dim=10, training_num=9000, test_num=1000, noise=0.01, **args):
        super(DummyData, self).__init__(**args)
        self.name = "dummy"
        self.dims = dim
        self.training_num = training_num
        self.test_num = test_num
        self.noise = noise

    def LoadTrainData(self):
        super(DummyData, self).LoadTrainData()
        self.training_instances = np.random.rand(self.training_num, self.dims)
        self.training_labels = np.array(
            [int(round(np.average(instance) + (np.random.ranf(1) * 2 - 1) * self.noise)) for instance in self.training_instances]
        )

    def LoadTestData(self):
        super(DummyData, self).LoadTestData()
        self.test_instances = np.random.rand(self.test_num, self.dims)
        self.test_labels = np.array(
            [int(round(np.average(instance) + (np.random.ranf(1) * 2 - 1) * self.noise)) for instance in self.test_instances]
        )

    def GetDebugTable(self):
        headers, data = super(DummyData, self).GetDebugTable()
        addtional_headers = ["noise"]
        additonal_data = [self.noise]
        return headers + addtional_headers, data + additonal_data
