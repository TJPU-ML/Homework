import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
path ='IRIS.csv'
pdData = pd.read_csv(path,header = None ,names = ['X1','Y1','admin'])
pdData.head()
positive = pdData[pdData['admin'] == 1]
negative = pdData[pdData['admin'] == 0]
fix,ax = plt.subplots(figsize = (10,5))
ax.scatter(positive['X1'],positive['Y1'],s = 30,c = 'b',marker = 'o',label = 'setosa')
ax.scatter(negative['X1'],negative['Y1'],s = 30,c = 'r',marker = 'x',label = 'versicolor')
ax.legend()
ax.set_xlabel('X1 score')
ax.set_ylabel('Y1 socre')
plt.show()
def sigmoid(z):
    return 1 / (1+np.exp(-z))
    nums = np.arange(-10,10,step = 1)
    fig,ax = plt.subplots(figsize = (12,4))
    ax.plot(nums,sigmoid(nums),'r')
plt.show()
def model(X,theta):
    return sigmoid(np.dot(X,theta.T))
pdData.insert(0,'ones',1)
matrix = pdData.as_matrix()
cols = matrix.shape[1]
X = matrix[:,0:cols-1]#特征值
Y = matrix[:,cols-1:cols]#真实值
theta = np.zeros([1,3])
def cost(X,y,theta):
    left = np.multiply(-y,np.log(model(X,theta)))
    print(left[1:5])
    right = np.multiply(1-y,np.log(1-model(X,theta)))
    print(right[1:5])
    return np.sum(left-right)/(len(X))
def gradient(X,y,theta):
    grad = np.zeros.shape(theta.shape)
    error = np.multiply(model(X,theta) - y).ravel()
    for j in range (len(theta.ravel)):
        term = np.multiply(error,X[:,j])
        grad[0,j] = np.sum(term)/(len(X))
    return grad
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2
def stopCriterion(type,value,threshold):
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1]-value[-2])<threshold
    elif type == STOP_GRAD:
        np.linalg.norm(value) < threshold
#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    print('X',X[1:5])
    print('Y',Y[1:5])
    return X, y


import time


def descent(data, theta, batchsize, stopType, thresh, alpha):  # 数据，参数，梯度下降的类型，停止的类型，停止的边缘，学习率
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # 每一次梯度下降的数据
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(X, y, theta)]

    while True:
        grad = gradient(X[k:k + batchsize], y[k:k + batchsize], theta)  # 梯度下降
        k += batchsize
        if k >= 100:  # 大于总数据
            k = 0
            X, y = shuffleData(data)  # 重新再来
        theta = theta - alpha * grad

        costs.append(cost(X, y, theta))

        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh):
            break
    return theta, i - 1, costs, grad, time.time() - init_time
def runExpe(data,theta,batchSize,stopType,thresh,alpha):
    n = 100
    theta,iter,costs,grad,dur = descent(data,theta,batchSize,stopType,thresh,alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==n: strDescType = "Gradient"
    elif batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta
result = runExpe(matrix, theta, 100,STOP_COST, thresh=0.000001, alpha=0.001)