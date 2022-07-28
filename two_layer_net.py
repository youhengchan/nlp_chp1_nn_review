# 实现层的类化以及正向传播
# 神经网络中有各种各样的层。我们将其视为 Python 的类
# 通过这种模块化。可以像搭建了搞积木一样构建网络
# 本书在实现这些层的时候，制定下面的代码规范：
# 1. 所有的层都有 forward() 方法和 backward() 方法
# 2. 所有的层都有 params 和 grads 实例变量

'''
代码规范说明：
首先, forward() 方法和 backward() 方法分别对应正向传播和反向传播。
其次，params 是用列表保存权重和偏置等参数（参数可能有多个，所以用列表保存）
grads 以与 params 中参数对应的形式，是用列表保存各个参数的梯度（后述）

因为在02节只考虑正向传播，所以我们仅关注代码规范中的以下两点：
1. 在层中实现 forward 方法
2. 将参数整理到实例变量 params 中

2022.7.28 0：59
到了第 4 节，补充完成了所有的反向传播，并添加了 SoftmaxWithLoss 的 Batch 实现
'''

# 导入 numpy, datetime, panda
import numpy as np  # numpy
import datetime     # time stamp


# 实现 Sigmoid 层
class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        return dout * self.y * (1 - self.y)


'''
Affine 层在初始化的时候接收权重和偏置。
此时 Affine 层的参数是权重和偏置（神经网络学习时，这两个参数随时被更新）
因此，我们使用列表将这两个参数保存在实例变量 params 中
然后实现基于 forward(x) 的正向传播的处理
'''
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        self.x = x
        W, b = self.params
        out = np.dot(x, W) + b
        return out

    def backward(self, dout):
        W, b = self.params
        dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, W.T)
        db = np.sum(dout, keepdims=False)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


# 支持单笔数据和 Batch 数据处理的 softmax 实现
def softmax_batch(x):
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    sum_exp = exp.sum(axis=-1, keepdims=True)
    return exp / sum_exp


# 支持单笔数据和 Batch 数据处理的 Cross Entropy 实现
def cross_entropy_batch(x, t):
    if x.ndim == 1:
        x = x.reshape((1, x.size))
        t = t.reshape((1, t.size))
    batch_size = x.shape[0]
    t = t.argmax(axis=1)
    avg_loss = -np.sum(np.log(x[np.arange(batch_size), t])) / batch_size
    return avg_loss


# Softmax 和 CrossEntropyLoss 合二为一的层
class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax_batch(x)
        self.t = t
        avg_loss = cross_entropy_batch(self.y, self.t)
        return avg_loss

    def backward(self, dout=1):  # 因为是网络最后一层，计算 loss，所以 dout 就是1，之后没有梯度再从更后面的层传回
        if self.y.ndim == 1:
            self.y = self.y.reshape((1, self.y.size))
            self.t = self.t.reshape((1, self.t.size))
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size
        dx *= dout
        return dx


def get_time_stamp():
    time_now = datetime.datetime.now()  # 获取一个当前时间对象
    year = str(time_now.year)  # 获取当前年份, 并转化为字符串
    month = str(time_now.month)  # 获取当前月份, 并转化为字符串
    day = str(time_now.day)  # 获取当前日, 并转化为字符串
    hour = str(time_now.hour)  # 获取当前小时, 并转化为字符串
    minute = str(time_now.minute)  # 获取当前分钟, 并转化为字符串
    second = str(time_now.second)  # 获取当前秒, 并转化为字符串
    millisecond = str(time_now.microsecond)[0:3]  # 获取当前毫秒, 并转化为字符串
    microsecond = str(time_now.microsecond)  # 获取当前微秒, 并转化为字符串
    return "".join([year, "_", month, "_", day, "_", hour, "_", minute, "_", second])

'''
现在，使用上面实现的层来实现神经网络的推理处理
这里实现 X => Affine => Sigmoid => Affine => S => Softmax => CrossEntropyLoss

输入 X 经由 Affine 层，Sigmmoid 层 和 Affine 层后输出得分 S
将得分 S 输入 Softmax 得到概率 P，再将概率 P 和标签 t 输入 
CrossEntropyLoss 得到损失 L
我们将这个神经网络实现为 TwoLayerNet 的类，将主推理处理实现为
predict(x) 方法
'''
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 生成层
        self.layers = [  # 实例化三个层
            Affine(W1, b1),  # 线形变换，实例化层对象
            Sigmoid(),  # sigmoid 激活，实例化层对象
            Affine(W2, b2),  # 线形变换，实例化层对象
            SoftmaxWithLoss()  # Softmax 和 CrossEntropy 合二为一层
        ]

        # 将所有的权重和权重对应梯度整理到列表中
        self.params, self.grads = [], []
        # 将所有需要学习的参数都保存到 params 列表中
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x, t):
        # 逐层计算前向传播
        for index, layer in enumerate(self.layers):
            x = layer.forward(x) if index != len(self.layers) - 1 else layer.forward(x, t)
        loss = x
        return loss

    def backward(self, dout=1):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def predict(self, x):
        # 逐层计算前向传播
        for index, layer in enumerate(self.layers):
            x = layer.forward(x) if index != len(self.layers) - 1 else softmax_batch(x)
        return x

    def save_weights(self, epochs, loss):
        path = "./weights/epochs_" + str(epochs) + "_loss_" + str(loss) + "_" + get_time_stamp()
        # print(f"[save weights]\n {self.params}\n")
        weights = np.array(self.params, dtype=object)
        np.save(path, weights)

    def load_weight(self, path):
        weights = np.load(path, allow_pickle=True)
        # print(f"[load weights]\n {weights}\n")
        if weights is not None:
            self.params = weights
        self.layers[0].params = self.params[0:2]
        self.layers[2].params = self.params[2:]

def test_two_layer_net():
    x = np.random.randn(10, 2)
    t = np.array([([[1, 0], [0, 1]][np.random.choice(2, 1).item()]) for i in range(10)])

    model = TwoLayerNet(2, 4, 3)
    loss = model.forward(x, t)

    print(f"[x]:\n {x}\n [t]:\n {t}\n [loss]:\n {loss}\n")


def test_save_weights():
    model = TwoLayerNet(2, 10, 3)
    model.save_weights(10000, 0.032)


def test_load_weights():
    model = TwoLayerNet(2, 10, 3)
    print(f"[model.params before load]:\n {model.params}")
    model.load_weight('./weights/epochs_100000_loss_0.013_2022_7_28_22_16_59.npy')
    print(f"[model.params after load]:\n {model.params}\n")


def main():
    test_load_weights()


if __name__ == "__main__":
    main()