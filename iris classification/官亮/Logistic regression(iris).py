# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 使用sklearn的datasets加载iris数据集
iris = datasets.load_iris()
# 获取数据集
x = iris.data[:, :2]
y = (iris.target != 0) * 1
print(y)
# 画数据分布图
plt.figure(figsize=(10, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color='b', label='0')
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='r', label='1')
plt.legend();

# 分割训练集与测试集
from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

loss_arr = []
iter_num = 0
iter_arr = []


# print(loss_arr)

class Logistic:
    # 初始化学习率lr，初始化训练次数iter_num
    def __init__(self, lr=0.01, iter_num=100000):
        self.lr = lr
        self.iter_num = iter_num

    def __x_init(self, x_train):
        # 创建一个全1矩阵
        x_shape = np.ones((x_train.shape[0], 1))
        # 数据集x每行对应于一个实例，最后增加一列元素恒置为1，使用concatenate方法增加
        return np.concatenate((x_shape, x_train), axis=1)

    # sigmoid函数
    def __sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # 损失函数
    def __loss(self, h, y_train):
        return (-y_train * np.log(h) - (1 - y_train) * np.log(1 - h)).mean()

    def fit(self, x_train, y_train):
        x_train = self.__x_init(x_train)
        self.theta = np.zeros(x_train.shape[1])
    # 梯度下降更新theta
        for i in range(self.iter_num):
            z = np.dot(x_train, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(x_train.T, (h - y_train) / y_train.size)
            self.theta -= self.lr * gradient

            z = np.dot(x_train, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y_train)

            loss_arr.append(loss)

            if (i % 1000 == 0):
                print(f'loss: {loss} \t')

    # 预测概率
    def predict_prob(self,x_train):
        x_train = self.__x_init(x_train)
        return self.__sigmoid(np.dot(x_train,self.theta))

    def predict(self,x_train):
         return self.predict_prob(x_train).round()


model = Logistic(lr=0.01, iter_num=5000)

model.fit(x_train, y_train)

preds = model.predict(x_test)
(preds == y_test).mean()

print((preds == y_test).mean())


plt.figure(figsize=(10, 6))
for i in range(5000):
    iter_arr.append(i)

x = iter_arr
y = loss_arr
plt.plot(x, y)
plt.show()