"""Data set (feature and label) class for Perceptron
"""
import numpy as np


class DataSet():
    def __init__(self, x, y):
        self.x = np.array(x)  # feature
        self.y = y  # label

    def size(self):
        return len(self.x)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"<{self.x}, {self.y}>"
