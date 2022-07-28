# coding: utf-8

import sys
sys.path.append('..')

from dataset import spiral
import matplotlib.pyplot as plt


def main():
    x, t = spiral.load_data()
    print('x', x.shape)
    print('t', t.shape)
    xx = x[:,0]
    yy = x[:,1]
    colors = ['red' for i in range(100)] + ['blue' for i in range(100)] + ['green' for i in range(100)]
    plt.scatter(xx, yy, c=colors)
    plt.title("Spiral Dataset Generated")
    plt.show()


if __name__ == "__main__":
    main()