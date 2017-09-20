from __future__ import absolute_import


class Model(object):
    def __init__(self, data, silent=False):
        self.name = 'base'
        self.data = data
        self.silent = silent
        self.metrics = {}

    def Train(self):
        pass

    def Eval(self):
        pass

    def Predict(self, instances):
        pass

    def GetDebugTable(self):
        dataset_header, dataset_data = self.data.GetDebugTable(
        ) if self.data else ([], [])
        return dataset_header + ["model_naame"] + self.metrics.keys(
        ), dataset_data + [self.name] + self.metrics.values()
