# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os


# 【文件状态：已完成，测试通过】

# 【文件说明】
# 用于绘图
# 1. 监控训练过程中的 LOSS 变化情况（训练每20轮自动生成并保存一次）
# 2. 保存训练结果图到本地

class Plot:
    def __init__(self, interval=20, max_iters=10):
        self.img_prefix = "./figure/"  # Loss 图像默认存在 figure 文件夹下
        self.img_format = ".png"       # Loss 图像默认存 png 格式
        self.interval = interval       # 默认 10 轮自动记录一次 Loss
        self.tmp_loss_list = []        # 存储训练中记录到的 Loss
        self.avg_loss_list = []        # average loss of tmp_loss_list
        self.max_iters = max_iters     # max training iteration in each epoch

    def get_truncated_latest_avg_loss(self):
        return round(self.avg_loss_list[-1], 3) if len(self.avg_loss_list) != 0 else 0

    def report(self, epoch, iter):
        if len(self.tmp_loss_list) == 0:
            print(f"[epoch] {epoch+1} | [iter] {iter+1} / {self.max_iters} | [loss] {self.avg_loss_list[-1]} ")

    def record(self, iter, loss):
        self.tmp_loss_list.append(loss)
        if len(self.tmp_loss_list) % self.interval == 0:  # 如果符合间隔要求
            self.avg_loss_list.append(np.average(self.tmp_loss_list))
            self.tmp_loss_list.clear()

    def save_loss(self, figure_name, append_time_stamp=True, ylim=None):
        x = np.arange(len(self.avg_loss_list))  # 所有记录的 Loss
        if ylim is not None:  # 如果设置了 y 的范围 ylim
            plt.ylim(*ylim)  # 设置绘图 y 轴范围为 ylim
        plt.plot(x, self.avg_loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.interval) + ')')
        plt.ylabel('loss')
        # 保存结果到本地
        figure_full_path = os.path.join(self.img_prefix, figure_name + self.img_format)
        if append_time_stamp:
            figure_full_path = os.path.join(self.img_prefix, figure_name + self.get_time_stamp() + self.img_format)
        plt.savefig(figure_full_path)
        # 运行时展示
        plt.show()

    def get_time_stamp(self):
        time_now = datetime.datetime.now()  # 获取一个当前时间对象
        year = str(time_now.year)  # 获取当前年份, 并转化为字符串
        month = str(time_now.month)  # 获取当前月份, 并转化为字符串
        day = str(time_now.day)  # 获取当前日, 并转化为字符串
        hour = str(time_now.hour)  # 获取当前小时, 并转化为字符串
        minute = str(time_now.minute)  # 获取当前分钟, 并转化为字符串
        second = str(time_now.second)  # 获取当前秒, 并转化为字符串
        millisecond = str(time_now.microsecond)[0:3]  # 获取当前毫秒, 并转化为字符串
        microsecond = str(time_now.microsecond)  # 获取当前微秒, 并转化为字符串
        return "".join(['_' + year, "_", month, "_", day, "_", hour, "_", minute, "_", second])


def test():
    plot = Plot()
    # 模拟 1000 步训练
    epsilon = 1e-3
    for learning_step in range(1000):
        loss = abs(1.732 * (1 - learning_step / 1000.0) + np.random.randn(1) / 10)
        plot.record(learning_step, loss)
        if loss <= epsilon:
            break
    # 结束训练，将训练结果以文件存到 ./figure 目录中
    plot.save_loss('test')


if __name__ == "__main__":
    test()

'''
测试输出：
见 ./figure/testxxx.png
'''