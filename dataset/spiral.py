# coding: utf-8
import numpy as np


def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 各类的样本数
    DIM = 2  # 数据的元素个数
    CLS_NUM = 3  # 类别数

    x = np.zeros((N*CLS_NUM, DIM)) # (300, 2)
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int) # (300, 3)

    for j in range(CLS_NUM): # 依次生成三类数据中的每一类数据点
        for i in range(N):#N*j, N*(j+1)): # 依次生成该类别中的每一个样本，每类生成100个
            rate = i / N  # 比例 rate 从 0 到 0.99
            radius = 1.0 * rate # 半径 radius 从 0 到 0.99
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2 # 根据类别和生成次序，构造 theta 参数，类别数越大，生成编号越大，相应 theta 也越大

            ix = N*j + i # 从 0 到 299
            x[ix] = np.array([radius*np.sin(theta), # 对 x 的一行进行赋值，计算的值是
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1 # 将对应类别位置置1

    return x, t