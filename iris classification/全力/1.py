# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:24:43 2018

@author: 全力
"""
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris   #从sklearn中导入鸢尾花数据

iris = load_iris()   #特征矩阵

#将原始数据集划分成训练集与测试集
speace_x = iris.data[0:100,0:2]
speace_y = iris.target[0:100,]
# 用train_test_split将数据按照7：3的比例分割训练集与测试集，
# 随机种子设为1（每次得到一样的随机数），设为0或不设（每次随机数都不同）
x_train, x_test, y_train,y_test = train_test_split(speace_x,speace_y,test_size = 0.3,random_state = 0)
#切片后数据的查看
#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)
x_train=np.insert(x_train,0,values=1, axis=1)
x_test=np.insert(x_test,0,values=1,axis=1)
#print(x_train )
theta = np.zeros([1,3])
#sigmoid函数   映射到概率
def sigmoid(z):
    return 1/(1+np.exp(-z))

#返回预测结果值
def model(X, theta):
    return sigmoid(np.dot(X,theta.T))



#损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))
#print(cost(x, y, theta))

#gradient : 计算每个参数的梯度方向
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    e = (model(X, theta)- y)
    print(np.shape(e))
    for j in range(len(theta.ravel())): 
        term = np.multiply(e, X[:,j])
        grad[0, j] = np.sum(term) / len(X)

    return grad


def descent(theta, batchSize, thresh, alpha):
    #梯度下降求解

    i = 0 # 迭代次数
    k = 0 # batch
    X = x_train
    y = y_train
    grad = np.zeros(theta.shape) # 计算的梯度
    costs = [cost(X, y, theta)] # 损失值
    while True:
        grad = gradient(X[k:k+batchSize,:], y[k:k+batchSize], theta) #batchSize为指定的梯度下降策略
        k += batchSize #取batch数量个数据
        if k >= len(x_train): 
            k = 0 
        theta = theta - alpha*grad # 参数更新
        costs.append(cost(X, y, theta)) # 计算新的损失
        i += 1 
        value = costs
        if abs(value[-1]-value[-2]) < thresh: #按损失函数是否改变停止
            break

    return theta, costs, grad


def runExpe(theta, batchSize, thresh, alpha):
    theta, costs, grad = descent(theta, batchSize, thresh, alpha)
    return theta

batchSize = len(x_train)    
runExpe(theta, batchSize, thresh=0.00001, alpha=0.001)
#print(theta)
#print(costs)
#print(grad)

def predict(X, theta):
    p = model(X,theta)
    if p>=0.5:
        return 1
    else:
        return 0
scaled_X = x_test
y = y_test
predictions = predict(scaled_X, theta)
count=0
for (a,b) in zip(predictions,y):
    if a==b:
        count=count+1
accuracy = (count/y.shape[0])*100
print ('{0}个训练数据的训练正确率为 {1}%'.format(scaled_X.shape[0],accuracy))





