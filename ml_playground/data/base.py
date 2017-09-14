from __future__ import absolute_import

import numpy as np


class Data(object):
    def __init__(self, silent=False, split_data=False, ratio=0.8, sematic_label=False):
        self.name = "base"
        self.dims = 0
        self.instance_num = 0
        self.instances = np.zeros(0)
        self.labels = np.zeros(0)
        self.training_num = 0
        self.training_instances = np.zeros(0)
        self.training_labels = np.zeros(0)
        self.training_one_hot_labels = np.zeros(0)
        self.test_num = 0
        self.test_instances = np.zeros(0)
        self.test_labels = np.zeros(0)
        self.test_one_hot_labels = np.zeros(0)
        self.max_label = 0
        self.one_hot_label_generated = False
        self.current_index = 0
        self.randomized_indexes = np.zeros(0)
        self.silent = silent
        self.split_data = split_data
        self.ratio = ratio
        self.sematic_label = sematic_label
        self.sematic_label_mapping = {}

    def Init(self):
        if self.split_data:
            self.LoadData()
            self.SplitTrainingAndTest(self.ratio)
        else:
            self.LoadTrainData()
            self.LoadTestData()
        self.__generate_one_hot_lables()

    def LoadData(self):
        pass

    def SplitTrainingAndTest(self, ratio, shuffle=True):
        self.training_num = int(self.instance_num * ratio)
        self.test_num = self.instance_num - self.training_num

        idx = range(self.instance_num)
        if shuffle:
            np.random.shuffle(idx)
        training_instances = []
        test_instances = []
        for index in range(self.training_num):
            training_instances.append(self.instances[idx[index]])
            self.training_labels = np.append(self.training_labels, self.labels[idx[index]])
        self.training_instances = np.asarray(training_instances)
        self.training_labels = self.training_labels.astype(int)
        for index in range(self.test_num):
            test_instances.append(self.instances[idx[self.training_num + index]])
            self.test_labels = np.append(self.test_labels, self.labels[idx[self.training_num + index]])
        self.test_instances = np.asarray(test_instances)
        self.test_labels = self.test_labels.astype(int)

    def LoadTrainData(self):
        pass

    def LoadTestData(self):
        pass

    def GetNextBatch(self, num):
        if num > self.training_num:
            num = self.training_num

        if self.current_index + num > len(self.randomized_indexes):
            self.randomized_indexes = np.arange(0, self.training_num)
            np.random.shuffle(self.randomized_indexes)
            self.current_index = 0

        dataShuffled = [self.training_instances[i] for i in self.randomized_indexes[self.current_index: self.current_index + num]]
        labelsShuffled = [self.training_labels[i] for i in self.randomized_indexes[self.current_index: self.current_index + num]]
        if self.one_hot_label_generated:
            oneHotLabelsShuffled = [
                self.training_one_hot_labels[i] for i in self.randomized_indexes[self.current_index: self.current_index + num]
            ]

        self.current_index += num

        return \
            np.asarray(dataShuffled), \
            np.asarray(labelsShuffled), \
            np.asarray(oneHotLabelsShuffled) if self.one_hot_label_generated else None

    def __generate_one_hot_lables(self):
        self.max_label = int(max(self.training_labels))

        self.training_one_hot_labels = np.zeros(
            self.training_num * (self.max_label + 1)
        ).reshape(self.training_num, self.max_label + 1)
        for index in range(self.training_num):
            self.training_one_hot_labels[index][self.training_labels[index]] = 1

        self.test_one_hot_labels = np.zeros(self.test_num * (self.max_label + 1)).reshape(self.test_num, self.max_label + 1)
        for index in range(self.test_num):
            if self.test_labels[index] > self.max_label:
                raise Exception("Test label are not seen in traning data %d" % self.test_labels[index])
            else:
                self.test_one_hot_labels[index][self.test_labels[index]] = 1

        self.one_hot_label_generated = True

    def GetDebugTable(self):
        return ["dataset_name", "dims", "training_num", "test_num", "training_labels"], \
            [self.name, self.dims, self.training_num, self.test_num, np.unique(self.training_labels)]


class SplitableData(Data):
    def __init__(self, **args):
        super(SplitableData, self).__init__(split_data=True, **args)
