#!/usr/bin/env python
#-*- encoding:utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#载入数据
iris = load_iris()
X = iris.data[:100]
y = iris.target[:100]

def std_data(datas):  #数据归一化
    means = datas.mean(axis=0)
    stds = datas.std(axis=0)
    N, M = datas.shape[0], datas.shape[1] + 1  #N：样本个数，M：维度
    data = np.ones((N, M))   #初始化 生成矩阵M*N
    data[:, 1:] = (datas - means) / stds
    return data

def gradAscent(X_train,y_train):  #梯度下降法
    ks = list(set(y_train))
    K = np.shape(list(set(y_train)))[0]
    N, M = X_train.shape[0], X_train.shape[1] + 1  #N是样本数，M是参数向量的维
    data = std_data(X_train)
    W = np.zeros((K - 1, M))  #存储参数矩阵
    priorEs = np.array([1.0 / N * np.sum(data[y_train == ks[i]], axis=0) for i in range(K - 1)])
    #各个属性的先验先验期望值
    loss_list=[]
    for it in range(1000):
        wx = np.exp(np.dot(W, data.transpose()))
        probs = np.divide(wx, 1 + np.sum(wx, axis=0).transpose())
        pEs = 1.0 / N * np.dot(probs, data)
        loss_list.append(np.sum(pEs - priorEs))
        gradient = pEs - priorEs + 1.0 / 1000 * W  #梯度，最后一项是防止过拟合
        W -= gradient  #对参数进行修正
    #损失函数图像
    xx = [i for i in range(1000)]
    fig, ax = plt.subplots()
    ax.plot(xx, loss_list, 'r-')
    ax.set_xlabel('item')
    ax.set_ylabel('loss')
    ax.set_title("loss of line")
    plt.show()
    return W,K
def LogisticRegression(W,K,X_test):  #线性回归
    N1, M1 = X_test.shape[0], X_test.shape[1] + 1   #N是样本数，M是参数向量的维
    data1=std_data(X_test)
    prob = np.ones((N1,K))
    prob[:,:-1] = np.exp(np.dot(data1,W.transpose()))
    prob /= np.array([np.sum(prob,axis = 1)]).transpose() #得到概率

    return prob

def predict(P):
    y_pred=[]
    for data in P:
        line=list(data)
        y_pred.append(line.index(max(line)))
    return y_pred

if __name__ == "__main__":
    split_list=[0.5,0.3,0.1]
    for i in split_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
        W,K=gradAscent(X_train, y_train)
        prob=LogisticRegression(W,K,X_test)
        y_pre=predict(prob)
        print("测试集所占百分比:{} 准确率:{}".format(i,accuracy_score(y_pre,y_test)))