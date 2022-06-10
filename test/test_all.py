from src.perceptron import *
import pytest
from math import *
import numpy as np


perceptron = Perceptron()


def test_no_offset():
    ds_list = DataList()
    ds_list.append([-1, -1], 1)
    ds_list.append([1, 0], -1)
    ds_list.append([-1, 1.5], 1)

    t = perceptron.perceptron_no_offset(ds_list)
    assert np.array_equal([-2,.5],t)

def test_no_offset_reordered():
    ds_list = DataList()
    ds_list.append([1, 0], -1)
    ds_list.append([-1, 1.5], 1)
    ds_list.append([-1, -1], 1)

    t = perceptron.perceptron_no_offset(ds_list)
    assert np.array_equal([-1,0],t)

def test_no_offset_0():
    ds_list = DataList()
    ds_list.append([-1, -1], 1)
    ds_list.append([1, 0], -1)
    ds_list.append([-1, 10], 1)

    t = perceptron.perceptron_no_offset(ds_list)
    assert np.array_equal([-6,5],t)

def test_no_offset_reordered_0():
    ds_list = DataList()
    ds_list.append([1, 0], -1)
    ds_list.append([-1, 10], 1)
    ds_list.append([-1, -1], 1)

    t = perceptron.perceptron_no_offset(ds_list)
    assert np.array_equal([-1,0],t)

def test_perceptron_zero_init():
    ds_list = DataList()
    ds_list.append([-4, 2], 1)
    ds_list.append([-2, 1], 1)
    ds_list.append([-1, -1], -1)
    ds_list.append([2, 2], -1)
    ds_list.append([1, -2], -1)

    [t,t0] = perceptron.perceptron(ds_list)
    assert np.array_equal([-3,3],t)
    assert np.array_equal(-3,t0)

def test_perceptron():
    ds_list = DataList()
    ds_list.append([-4, 2], 1)
    ds_list.append([-2, 1], 1)
    ds_list.append([-1, -1], -1)
    ds_list.append([2, 2], -1)
    ds_list.append([1, -2], -1)

    [t,t0] = perceptron.perceptron(ds_list,np.array([-3,3]),-3)
    assert np.array_equal([-3,3],t)
    assert np.array_equal(-3,t0)


