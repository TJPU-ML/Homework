# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:41:52 2018

@author: 777
"""
#导入相关的包
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


#对数几率函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#代入对数几率函数后的预测函数
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))
#损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))  # 左边的连乘
    right = np.multiply((1 - y), np.log(1 - model(X, theta))) # 右边的连乘
    return np.sum(left - right) / (len(X))

def gradient(X, y, theta):
    # 求解梯度 grad为 theta梯度的更新值
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):

        temp = np.multiply(error, X[:,j])
        grad[0, j] = np.sum(temp) / len(X)

    return grad

# 设定三种停止策略 分别是按迭代次数、按损失函数的变化量、按梯度的变化量
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

# threshold为指定阈值
def stopCriterion(stype, value, threshold):
    #设定三种不同的停止策略
    if stype == STOP_ITER:        
        return value > threshold # 按迭代次数停止

    elif stype == STOP_COST:
        return abs(value[-1]-value[-2]) < threshold

    elif stype == STOP_GRAD:
        return np.linalg.norm(value) < threshold 
    
    # 打乱训练集
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降法
    # batchSize：为1代表随机梯度下降
    # stopType 停止策略类型
    # thresh 阈值
    # alpha 学习率

    init_time = time.time() 
    i = 0 # 迭代次数
    k = 0 # batch 迭代数据的初始量
    X, y = shuffleData(data)
    grad = np.zeros(np.shape(theta)) # 计算的梯度
    costs = [cost(X, y, theta)] # 损失值

    while True:
        # batchSize为指定的梯度下降策略 
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize #取batch数量个数据
        if k >= n: 
            k = 0 
            X, y = shuffleData(data) #打乱数据
        theta = theta - alpha*grad # 参数更新
        costs.append(cost(X, y, theta)) # 更新损失函数
        i += 1 

        if stopType == STOP_ITER:       
            value = i

        elif stopType == STOP_COST:     
            value = costs

        elif stopType == STOP_GRAD:     
            value = grad

        if stopCriterion(stopType, value, thresh): 
            break

    return theta, i-1, costs, grad, time.time() - init_time

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    # 损失率与迭代次数的展示函数
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)

    if batchSize == n: 
        strDescType = "Gradient"

    elif batchSize == 1:  
        strDescType = "Stochastic"

    else: 
        strDescType = "Mini-batch ({})".format(batchSize)

    name += strDescType + " descent - Stop: "

    if stopType == STOP_ITER: 
        strStop = "{} iterations".format(thresh)

    elif stopType == STOP_COST: 
        strStop = "costs change < {}".format(thresh)

    else: 
        strStop = "gradient norm < {}".format(thresh)

    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')

    return theta

# 载入训练集
path = 'data' + os.sep + 'iris70.txt'
pdData = pd.read_csv(path, header=None, names=['length', 'width', 'Admitted'])
pdData.insert(0, 'Ones', 1)
orig_data = pdData.as_matrix() 
X = orig_data[:,:3]
y = orig_data[:,-1]

theta = np.zeros([1, 3])
# 设置迭代次数为 n次结束
n=100
runExpe(orig_data, theta, n, STOP_ITER, thresh=10000, alpha=0.001)

#预测函数
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]
#测试集代入，输出正确率
ath = 'data' + os.sep + 'iris70tr.txt'
dData = pd.read_csv(ath, header=None, names=['length', 'width', 'Admitted'])
dData.insert(0, 'Ones', 1) 
scaled_data=dData.as_matrix() 
scaled_X = scaled_data[:,:3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
