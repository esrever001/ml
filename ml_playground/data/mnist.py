from __future__ import absolute_import

from ml_playground.data.base import Data
from ml_playground.utils.data import ConvertByteDataToNum, Standardized
from ml_playground.utils.visual import PrintProgress
import numpy as np
from PIL import Image


class MNIST(Data):

    TRAINING_INSTANCE_FILE = "ml_playground/data/dataset/mnist/train-images-idx3-ubyte"
    TRAINING_LABEL_FILE = "ml_playground/data/dataset/mnist/train-labels-idx1-ubyte"
    TEST_INSTANCE_FILE = "ml_playground/data/dataset/mnist/t10k-images-idx3-ubyte"
    TEST_LABEL_FILE = "ml_playground/data/dataset/mnist/t10k-labels-idx1-ubyte"

    def __init__(self, **args):
        super(MNIST, self).__init__(**args)
        self.name = "MNIST"
        self.dims = 784
        self.training_num = 60000
        self.test_num = 10000

    def LoadTrainData(self):
        super(MNIST, self).LoadTrainData()
        self.training_num, self.training_instances = self.__readInstanceFile(self.TRAINING_INSTANCE_FILE)
        _, self.training_labels = self.__readLabelFile(self.TRAINING_LABEL_FILE)

    def LoadTestData(self):
        super(MNIST, self).LoadTestData()
        self.test_num, self.test_instances = self.__readInstanceFile(self.TEST_INSTANCE_FILE)
        _, self.test_labels = self.__readLabelFile(self.TEST_LABEL_FILE)

    def Init(self):
        super(MNIST, self).Init()
        self.training_instances, self.test_instances = Standardized(self.training_instances, self.test_instances)

    def __readLabelFile(self, filename):
        input_file = open(filename, "rb")
        magic_number = ConvertByteDataToNum(input_file.read(4))
        assert (magic_number == 2049)
        total_cnt = ConvertByteDataToNum(input_file.read(4))
        labels = range(total_cnt)

        for index in range(total_cnt):
            if not self.silent:
                PrintProgress(index + 1, total_cnt, "Loading labels")
            labels[index] = ord(input_file.read(1))

        return total_cnt, np.asarray(labels)

    def __readInstanceFile(self, filename):
        input_file = open(filename, "rb")
        magic_number = ConvertByteDataToNum(input_file.read(4))
        assert (magic_number == 2051)
        total_cnt = ConvertByteDataToNum(input_file.read(4))
        rows = ConvertByteDataToNum(input_file.read(4))
        cols = ConvertByteDataToNum(input_file.read(4))

        instances = range(total_cnt)

        for index in range(total_cnt):
            if not self.silent:
                PrintProgress(index + 1, total_cnt, "Loading instances")
            img = range(rows * cols)
            for r in range(rows):
                for c in range(cols):
                    img[r * cols + c] = ord(input_file.read(1))
            instances[index] = img

        return total_cnt, np.asarray(instances)

    def DebugTrainingInstance(self, index):
        if index >= self.training_num:
            return
        super(MNIST, self).DebugTrainingInstance(index)
        img = Image.new('L', (28, 28))
        instance = [int(element * 255.0) for element in self.raw_training_instances[index].tolist()]
        img.putdata(instance)
        img.show()
