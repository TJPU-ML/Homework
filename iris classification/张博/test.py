# !/usr/bin/env python3
# encoding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

#加载数据集
iris=sns.load_dataset("iris")
iris.head()
# print(iris.head())

# 对类型进行one-hot编码(使用get_dummies进行one-hot编码)
dummies_iris = pd.get_dummies(iris['species'], prefix= 'species')
iris_df = pd.concat([iris, dummies_iris], axis=1)
iris_df.drop(['species'], axis=1, inplace=True)
iris_df.describe()
# print(iris_df.describe())

#模型建立
def sigmoid(X):
    return 1/(1+np.exp(-X))

def gra_ascent_train(X, Y, alpha=0.01, epoch=100, test_size=0.3, random_state=1):
    """
    梯度下降法
    :param X: 所要划分的样本特征集样本
    :param Y: 所要划分的样本结果
    :param alpha:学习效率0.01
    :param epoch:迭代次数
    :param test_size: 样本占比
    :param random_state: 随机数种子
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)   # 分割数据集
    w = np.ones((X_train.shape[1], Y_train.shape[1]))
    error_train = np.ones((epoch,))
    error_test = np.ones((epoch,))
    for i in range(epoch):
        error_train[i], error_test[i] = predict_error(
            X_train, X_test, Y_train, Y_test, w)  # 记录
        error = predict(X_train, w) - Y_train  # 更新权值
        w = w - alpha * X_train.transpose() * error
    return w, error_train, error_test

# 预测
def predict(X_train,w):
    return sigmoid( X_train * w )

# 获取预测错误率
def predict_error(X_train,X_test,Y_train,Y_test,w):
    train_pred = np.argmax(predict(X_train,w), axis=1)
    test_pred = np.argmax(predict(X_test,w), axis=1)
    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    error_train = 1- metrics.accuracy_score(Y_train, train_pred)
    error_test = 1- metrics.accuracy_score(Y_test, test_pred)
    return error_train,error_test

def result():
    Y = iris_df.iloc[:, 4:].values
    X = iris_df.iloc[:, :5].values
    X[:, 4] = 1
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, test_sizei in enumerate([0.1, 0.3, 0.5]):
        w, error_train, error_test = gra_ascent_train(
            np.mat(X), np.mat(Y), alpha=0.001, epoch=5000, test_size=test_sizei, random_state=1)
        print("test_size:{0:.1f}|  accuracy_train: {1:.3f}  accuracy_test:{2:.3f}".format(
            test_sizei, 1 - error_train[-1:][0],
                        1 - error_test[-1:][0]))
        # 绘图
        sns.lineplot(x=range(0, len(error_test), 1), y=error_test,
                     linewidth=1.5, ax=axs[i])
        sns.lineplot(x=range(0, len(error_train), 1), y=error_train,
                     linewidth=1.5, ax=axs[i])
        axs[i].set_xlabel("training_number")
        axs[i].set_ylabel("error_rate")
        axs[i].set_title("test_set_size:{0:.1f}".format(test_sizei))
        plt.legend(['test_set', 'training_set'])
    plt.show()

if __name__ == '__main__':
    result()
