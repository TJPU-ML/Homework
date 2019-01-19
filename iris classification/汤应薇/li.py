# -*- coding: utf-8 -*
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import  preprocessing
from numpy import *
def load_data(bili):

    X_all = iris.data
    y_all = iris.target
    #删除第三种数据
    y_all = y_all.tolist()
    p = [i for i in range(len(y_all)) if y_all[i] == 2]
    X_all = np.delete(X_all, p, axis=0)
    y_all = np.delete(y_all, p, axis=0)

    # 随机分层切分数据集，并转换成矩阵
    X_test, X_train, y_test, y_train = train_test_split(X_all, y_all, test_size=bili, stratify=y_all)  # stratify分层

    #标准化数据
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    #训练集以及测试集进行赋值
    x_1 = X_train[:, 0]
    x_2 = X_train[:, 1]

    x_3 = X_test[:, 0]
    x_4 = X_test[:, 1]
    # 转变为行向量
    x_1 = x_1.reshape(len(x_1), 1)
    x_2 = x_2.reshape(len(x_2), 1)

    x_3 = x_3.reshape(len(x_3), 1)
    x_4 = x_4.reshape(len(x_4), 1)
    # 返回训练样本个数
    N = len(x_1)
    # 返回测试样本的个数
    M = len(x_3)
    y_train = y_train.reshape(len(y_train), 1)
    # 变成行向量
    return x_1, x_2, x_3, x_4, y_train, y_test, N, M

#激活函数
def sig(x):
    return (1 / (1 + np.exp(-x)))

#梯度下降求得最优
def tidu(x_1, x_2, x_3, x_4, y_train, y_test, N, M, q0, q1, q2, index, sunshi):
#变量名 cishu为训练次数 buc为训练步长,权重矩阵q初始值设为1,初始计数index为0,损失值赋初值空
        cishu = 10000
        buc = 0.01
        while (index < cishu):
            y = q0 + q1 * x_1 + q2 * x_2
            y = sig(y)
            # 预测与实际差值
            y_c = y-y_train
            # 损失值计算
            sun = (- np.dot(np.transpose(y_train), np.log(y)) - np.dot(np.transpose(1 - y_train), np.log(1 - y))) / N
            sunshi.append(sun)

            # 计算乘积，直接点乘，然后求和取平均(步长太大，损失函数下降太快考虑求平均来看)
            j0 = sum(y_c)/N
            j1 = sum(x_1 * y_c)/N
            j2 = sum(x_2 * y_c)/N
            # 更新权重值
            q0 = q0 - buc * j0
            q1 = q1 - buc * j1
            q2 = q2 - buc * j2
            # 迭代次数计算累计
            index += 1
        #删除多余的权重矩阵
        q0 = q0.tolist()
        s = list(range(M, N))
        q0 = np.delete(q0, s)
        q1 = np.delete(q1, s)
        q2 = np.delete(q2, s)
        q0 = q0.reshape(M, 1)
        q1 = q1.reshape(M, 1)
        q2 = q2.reshape(M, 1)
        # 将训练得到的参数代入，计算得到预测值
        y_ce = q0 + q1 * x_3 + q2 * x_4
        y_ce = sig(y_ce)
        # 阈值设为0.5，大于或等于0.5则预测为1，小于0.5则为0
        y_zl = []
        for val in y_ce:
            if (val >= 0.5):
                y_zl.append(1)
            else:
                y_zl.append(0)
        # 判断类别并输出准确率
        print('zhunquedu: ', accuracy_score(y_test, y_zl))
        #得到损失值画出随迭代次数变化的损失函数变化曲线
        sunshi = np.array(sunshi)
        sunshi = sunshi.reshape(cishu, 1)
        cd = range(len(sunshi))
        plt.xlabel('epochs')
        plt.ylabel('Training loss')
        plt.plot(cd, sunshi)
        plt.savefig('loss5.png')
        plt.show()

        return q0, q1, q2
#画出决策边界
def plot_bianjie(x_3, x_4, y_test, q0, q1, q2):
        # 所需属性数据
        type1_x = []
        type1_y = []
        type2_x = []
        type2_y = []

        for i in range(M):
            if y_test[i] == 0:
                type1_x.append(x_3[i])
                type1_y.append(x_4[i])
            if y_test[i] == 1:
                type2_x.append(x_3[i])
                type2_y.append(x_4[i])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 数据绘图   
        type1 = ax.scatter(type1_x, type1_y, s=30, c='g')
        type2 = ax.scatter(type2_x, type2_y, s=30, c='r')
        x = np.arange(-2.00, 2.00, 0.01)
        #得到根据y与x之间的关系画出决策边界
        y = (-q0 - q1 * x) / q2
        ax.plot(x, y)
        # 设置横坐标和纵坐标
        ax.set_xlabel("Sepal length")
        ax.set_ylabel("Sepal width")
        ax.legend((type1, type2), (u'setosa', u'versicolor'), loc='best')
        plt.savefig('juece.png')
        plt.show()

if __name__ == '__main__':
    # 从sklearn的datasets中导入训练数据
    iris = datasets.load_iris()
    bili = 0.5
    load_data(bili)
    # 自行更改比例值
    sunshi=[]
    index = 0
    x_1, x_2, x_3, x_4, y_train, y_test, N, M = load_data(bili)
    q0 = np.zeros((N, 1))
    q1 = np.zeros((N, 1))
    q2 = np.zeros((N, 1))
    q0, q1, q2 = tidu(x_1, x_2, x_3, x_4, y_train, y_test, N, M, q0, q1, q2, index, sunshi)
    plot_bianjie(x_3, x_4, y_test,  q0=q0[0], q1=q1[0], q2=q2[0])