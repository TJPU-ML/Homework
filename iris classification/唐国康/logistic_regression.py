#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plot
from iris import iris_data
import matplotlib.patches as mpatches


# 以特定步长,  直接迭代很多次,  最后 W 会在最小值附近
def cal_W(X, Y, alpha=0.001, max_iter=10000):
    W = np.mat(np.random.rand(3, 1))
    loss_func_values, iter_times = [], []
    for i in range(max_iter):
        H = 1 / (1 + np.exp(-X * W))
        dl_dw = X.T * (H - Y)
        W -= alpha * dl_dw
        if i % 100 == 0:
            value = -Y.T * np.exp(H) - (1-Y).T * np.exp(1-H)
            loss_func_values.append( value.tolist()[0][0] )
            iter_times.append(i)
    return W, loss_func_values, iter_times


# 设定测试集为 70%
train_set_amount = 70
train_set = iris_data[:int(train_set_amount/2*3 )] + iris_data[int(-train_set_amount/2*3):]
X = list(zip(train_set[::3], train_set[1::3]))
X = [[1, *_] for _ in X]
Y = train_set[2::3]
X, Y = np.mat(X), np.mat(Y).T


# 设定迭代次数,  迭代 W, 记录损失函数的值
max_iter = 20000
W, loss_func_vlaues, iter_times = cal_W(X, Y, max_iter=max_iter)


# 把数据集的点画上去
iter_patch = mpatches.Patch(label='iter {} times'.format(max_iter))
rate_patch = mpatches.Patch(label='train set: {}%'.format(train_set_amount))
plot.subplot(211)
plot.legend(handles=[iter_patch, rate_patch])
plot.scatter(X[:, 1][Y == 0].A, X[:, 2][Y == 0].A, marker='^')
plot.scatter(X[:, 1][Y == 1].A, X[:, 2][Y == 1].A)

# 画出类别分界线,  分界线上的点让 z = 0, 分界线上的点代入模型得到的概率值为 0.5
w0, w1, w2 = W.reshape(1, 3).tolist()[0]
plot_x1 = np.arange(4, 8)
plot_x2 = -w0/w2 - w1/w2*plot_x1
plot.plot(plot_x1, plot_x2)

# 画出损失函数的随着迭代次数的曲线
plot.subplot(212)
plot.plot(iter_times, loss_func_vlaues)
plot.show()
