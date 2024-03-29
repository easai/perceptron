"""DataList is a list of data sets
"""
from .dataset import DataSet


class DataList():
    def __init__(self):
        self.data_list = []

    def append(self, x, y):
        self.data_list.append(DataSet(x, y))

    def dim(self):
        dim = 0
        if self.data_list:
            dim = self.data_list[0].size()
        return dim

    def __str__(self) -> str:
        txt = ""
        first = True
        for item in self.data_list:
            if first:
                txt += ","
                first = False
            txt += item.__str__()
        return txt
