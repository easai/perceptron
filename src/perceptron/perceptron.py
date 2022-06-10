"""Perceptron class
"""
import numpy as np


class Perceptron():

    def agreement(self, x, y, t, t0):
        return y*(t.dot(x)+t0)

    def perceptron(self, data_list, t=np.array([0, 0]), t0=0):
        for i in range(5):
            for idx, item in enumerate(data_list.data_list):
                if self.agreement(item.x, item.y, t, t0) <= 0:
                    t = t+item.y*item.x
                    t0 = t0+item.y
                    print(f"{(idx+1)=} {t=} {t0=}")

        return [t,t0]

    def perceptron_no_offset(self, data_list, t=np.array([0])):
        t=np.zeros(data_list.dim())
        for i in range(5):
            for idx, item in enumerate(data_list.data_list):
                if self.agreement(item.x, item.y, t, 0) <= 0:
                    t = t+item.y*item.x
                    print(f"{(idx+1)=} {t=}")

        return t
