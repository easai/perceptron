from src.perceptron import *
import pytest
from math import *
import numpy as np


perceptron = Perceptron()


ds_list = DataList()
ds_list.append([1, 0, 1], 2)
ds_list.append([1, 1, 1], 2.7)
ds_list.append([1, 1, -1], -.7)
ds_list.append([-1, 1, 1], 2)
t = np.array([0, 1, 2])

r = perceptron.risk(ds_list, t)


def test_risk():
    r = perceptron.risk(ds_list, t)
    assert r == 1.25


def test_risk_sq():
    r = perceptron.risk_sq(ds_list, t)
    assert r == 0.1475
