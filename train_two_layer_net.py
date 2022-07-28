# coding: utf-8

import sys
sys.path.append('..')
import cupy as cp                      # cupy GPU
import numpy as np                     # numpy
from optimizer import SGDL             # optimizer
from dataset import spiral             # dataset
from two_layer_net import TwoLayerNet  # Model
from plot import Plot                  # Plot

def train():
    # Step1: Set hyper parameters
    max_epoch = 10000
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # Step2: Load dataset & Model & Optimizer & Plotter
    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGDL(learning_rate=learning_rate)

    data_size = len(x)  # 300
    max_iters = data_size // batch_size  # 300 // 30 = 10
    plot = Plot(interval=10, max_iters=max_iters)

    # Step3: Permutate dataset Randomly
    for epoch in range(max_epoch):
        idx = np.random.permutation(data_size)
        x = x[idx]  # reorder x using random permutation indexs
        t = t[idx]  # reorder t using random permutation indexes

        for iters in range(max_iters):
            batch_x = x[iters*batch_size: (iters+1)*batch_size]
            batch_t = t[iters*batch_size: (iters+1)*batch_size]

            # Step4: Forward() to calc loss, Backward() to calc grads, use optimizer to update parameters
            avg_loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            # Step5: Record and plot loss
            plot.record(iters, avg_loss)
            plot.report(epoch, iters)

    # Step6: Save model weights and loss figure
    model.save_weights(max_epoch, plot.get_truncated_latest_avg_loss())
    plot.save_loss("two_layer_net")
    print(f"[final weights]:\n {model.params}\n")

    # After Training, Predict
    x, t = spiral.load_data()
    result = model.predict(x)
    print(f"[raw result]:\n {result}\n")
    result = np.array(result)
    result = np.argmax(result, axis=-1)
    print(f"[argmax result]:\n {result}\n")
    # print(f"[t]:\n {t}\n")


def main():
    train()


if __name__ == "__main__":
    main()