"""Data set (feature and label) class for Perceptron
"""
import numpy as np


class DataSet():
    def __init__(self, x, y):
        self.x = np.array(x)  # feature
        self.y = y  # label

    def size(self):
        return len(self.x)
