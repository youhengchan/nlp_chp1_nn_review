# coding: utf-8

import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from dataset import spiral
import numpy as np


def predict():
    # Step1: Load Model & Weights
    model = TwoLayerNet(2, 10, 3)
    model.load_weight('/home/kiko/PycharmProjects/chp1_nn_review/weights/epochs_10000_loss_0.033_2022_7_28_23_43_48.npy')
    print(f"[loaded params]:\n {model.params}\n")

    # Step2: Load Dataset
    x, t = spiral.load_data()
    result = model.predict(x)
    result = np.array(result)
    result = np.argmax(result, axis=-1)
    print(f"[result]:\n {result}\n")

    xx = x[:, 0]
    yy = x[:, 1]
    colors = []
    for cls in result:
        if cls == 0:
            colors.append('pink')
        elif cls == 1:
            colors.append('purple')
        else:
            colors.append('brown')
    plt.scatter(xx, yy, c=colors)
    plt.title('Model Prediction Visualization')
    plt.show()
    plt.savefig('figure/predict.png')


def main():
    predict()


if __name__ == "__main__":
    main()