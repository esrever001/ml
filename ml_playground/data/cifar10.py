from __future__ import absolute_import

from ml_playground.data.base import SplitableData
from ml_playground.utils.data import Standardized
import numpy as np
import glob
import cPickle
from PIL import Image


class CIFAR10(SplitableData):

    INSTANCES_FILE = "ml_playground/data/dataset/cifar-10/data_batch_*"
    META_FILE = "ml_playground/data/dataset/cifar-10/batches.meta"

    def __init__(self, **args):
        super(CIFAR10, self).__init__(**args)
        self.name = "CIFAR-10"

    def LoadData(self):
        self.instances, self.labels = self.__readInstanceFile()
        self.instance_num, self.dims = self.instances.shape
        self.__readLabelMetaFile()

    def Init(self):
        super(CIFAR10, self).Init()
        self.training_instances, self.test_instances = Standardized(self.training_instances, self.test_instances)

    def __readInstanceFile(self):
        instances = None
        labels = None
        files = glob.glob(self.INSTANCES_FILE)
        for file in files:
            with open(file, 'rb') as fo:
                dict = cPickle.load(fo)
            data = dict['data']
            label = dict['labels']
            if instances is not None:
                instances = np.concatenate((instances, data), axis=0)
                labels = np.concatenate((labels, label), axis=0)
            else:
                instances = data
                labels = label
        for index in range(len(instances)):
            instances[index] = self.__rearrange_data(instances[index])
        return instances / 255.0, labels

    def __rearrange_data(self, input_instance):
        instance = np.zeros(input_instance.shape)
        for index in range(1024):
            instance[index * 3] = input_instance[index]
            instance[index * 3 + 1] = input_instance[index + 1024]
            instance[index * 3 + 2] = input_instance[index + 1024 * 2]
        return instance

    def __readLabelMetaFile(self):
        with open(self.META_FILE, 'rb') as fo:
            dict = cPickle.load(fo)
            label_names = dict['label_names']
        self.sematic_label = True
        for index in range(len(label_names)):
            self.sematic_label_mapping[index] = label_names[index]
            self.sematic_label_mapping[label_names[index]] = index

    def DebugTrainingInstance(self, index):
        if index >= self.training_num:
            return
        super(CIFAR10, self).DebugTrainingInstance(index)
        img = Image.new('RGB', (32, 32))
        instance = [int(element * 255.0) for element in self.raw_training_instances[index].tolist()]
        data = [0] * 1024
        for index in range(1024):
            data[index] = instance[index * 3] * 256 * 256 + instance[index * 3 + 1] * 256 + instance[index * 3 + 2]
        img.putdata(data)
        img.show()
