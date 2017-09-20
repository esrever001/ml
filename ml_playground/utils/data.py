from __future__ import absolute_import

import numpy as np


def ConvertByteDataToNum(byteStr):
    num = 0
    for byte in byteStr:
        num = num * 256 + ord(byte)
    return num


def Standardized(training_data, test_data):
    mean = np.mean(training_data, axis=0)
    stds = np.std(training_data, axis=0)
    non_zero_stds = np.vectorize(lambda x: x if x > 0 else 1.0)(stds)
    return (training_data - mean) / non_zero_stds, (
        test_data - mean) / non_zero_stds
