"""Perceptron class
"""
import numpy as np


class Perceptron():

    def agreement(self, x, y, t, t0):
        return y * (t.dot(x) + t0)

    def perceptron(self, data_list, t=np.array([0, 0]), t0=0):
        for i in range(5):
            for idx, item in enumerate(data_list.data_list):
                if self.agreement(item.x, item.y, t, t0) <= 0:
                    t = t + item.y * item.x
                    t0 = t0 + item.y
                    print(f"{(idx+1)=} {t=} {t0=}")

        return [t, t0]

    def perceptron_no_offset(self, data_list, t=np.array([0])):
        t = np.zeros(data_list.dim())
        for i in range(5):
            for idx, item in enumerate(data_list.data_list):
                if self.agreement(item.x, item.y, t, 0) <= 0:
                    t = t + item.y * item.x
                    print(f"{(idx+1)=} {t=}")

        return t

    def risk(self, data_list, t):
        r = []
        for idx, item in enumerate(data_list.data_list):
            res = 0
            z = item.y - t.dot(item.x)
            if z < 1:
                res = 1 - z
            r.append(res)
        print(f"{r=}")
        res = np.array(r).mean()
        print(f"{res=}")
        return res

    def risk_sq(self, data_list, t):
        r = []
        for idx, item in enumerate(data_list.data_list):
            z = item.y - t.dot(item.x)
            r.append(z * z / 2)
        print(f"{r=}")
        res = np.array(r).mean()
        print(f"{res=}")
        return res
