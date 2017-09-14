from __future__ import absolute_import

from ml_playground.data.base import SplitableData
from ml_playground.utils.data import Standardized
import numpy as np


class Iris(SplitableData):

    INSTANCE_FILE = "ml_playground/data/dataset/iris/iris.data"

    def __init__(self, **args):
        super(Iris, self).__init__(**args)
        self.name = "iris(UCI)"
        self.dims = 4
        self.instance_num = 150
        self.sematic_label = True
        self.sematic_label_mapping = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2,

            0: "Iris-setosa",
            1: "Iris-versicolor",
            2: "Iris-virginica",
        }

    def LoadData(self):
        self.instances, self.labels = self.__readFile(self.INSTANCE_FILE)

    def Init(self):
        super(Iris, self).Init()
        self.training_instances, self.test_instances = Standardized(self.training_instances, self.test_instances)

    def __readFile(self, filename):
        input_file = open(filename, "rb")
        instances = []
        labels = []
        for line in input_file:
            line = line[:-1]
            current = line.split(',')
            instance = [float(value) for value in current[0:4]]
            label = self.sematic_label_mapping[current[4]]
            instances.append(instance)
            labels.append(int(label))

        return np.asarray(instances, dtype=np.float64), np.asarray(labels, dtype=np.int)
