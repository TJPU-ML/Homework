import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
import numpy.random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

import time

pdData = pd.read_csv("IRIS.csv", header=None, names=['NO1', 'NO2', 'NO3', 'NO4', 'kind'])
pdData.head()
# def iris_type(s):
#     class_label={b'Iris-setosa':0,b'Iris-versicolor':1}
#     return class_label[s]
#
# filepath='IRIS.csv'  # 数据文件路径
# pdData=pd.read_csv(filepath,dtype=float,delimiter=',',converters={4:iris_type})

positive = pdData[
    pdData['kind'] == 1]
negative = pdData[
    pdData['kind'] == 0]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(positive['NO1'], positive['NO2'], s=30, c='b', marker='o', label='短')
ax.scatter(negative['NO1'], negative['NO2'], s=30, c='r', marker='x', label='长')
ax.legend()
ax.set_xlabel('NO1')
ax.set_ylabel('NO2')


# plt.show() //画图

#  映射到概率的函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 返回预测结果值
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


pdData.insert(0, 'Ones', 1)
# print(pdData)
orig_data = pdData.as_matrix()
cols = orig_data.shape[1]
X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]
theta = np.zeros([1, 5])


# print(X.shape,y.shape,theta.shape)

def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


# qqq = cost(X, y, theta)
# print(theta)

def gradient(X, y, theta):
    grad = np.zeros(theta.shape)  # （1,3）
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):  # for each parmeter
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)

    return grad


STOP_ITER = 0


def stopCriterion(type, value, threshold):
    if type == STOP_ITER:        return value > threshold


# 打乱
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值
    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1
        if stopType == STOP_ITER:
            value = i
        if stopCriterion(stopType, value, thresh): break

    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta

plt.show()


def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]


scaled_X = orig_data[:, :5]
y = orig_data[:, 5]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

new_data = shuffle(orig_data)
a = new_data[:50, :]
b = new_data[50:, :]
n = 100
theta1 =  runExpe(a, theta, n, STOP_ITER, thresh=5000, alpha=0.006)
plt.show()
# print(theta1)
# print(b)

# c = orig_data[52:53, :5]
# print(c.T)
# d = model(c,theta1)
# print(d)
# print(model(c,theta1))
f=50
g=51
h =50
j = 0

while f < 100:
    c = new_data[f:g, :5]
    d = model(c, theta1)
    i = new_data[f:g, 5:6]
    if (d<0.5 and i<0.5):
        j +=1
    elif (d>=0.5 and i >0.5):
        j+=1
    f += 1
    g += 1
print("取50%时正确率为",j/h)

new_data1 = shuffle(orig_data)
a11 = new_data1[:70, :]
b11 = new_data1[570:, :]
n11 = 100
theta11 =  runExpe(a11, theta, n, STOP_ITER, thresh=5000, alpha=0.006)
plt.show()
# print(theta1)
# print(b)

# c = orig_data[52:53, :5]
# print(c.T)
# d = model(c,theta1)
# print(d)
# print(model(c,theta1))
f11=70
g11=71
h11 =30
j11 = 0

while f11 < 100:
    c11 = new_data1[f11:g11, :5]
    d11 = model(c11, theta11)
    i11 = new_data1[f11:g11, 5:6]
    if (d11<0.5 and i11<0.5):
        j11 +=1
    elif (d11>=0.5 and i11 >0.5):
        j11+=1
    f11 += 1
    g11 += 1
print("取70%时正确率为",j11/h11)

new_data22 = shuffle(orig_data)
a = new_data[:90, :]
b = new_data[90:, :]
n = 100
theta1 =  runExpe(a, theta, n, STOP_ITER, thresh=5000, alpha=0.006)
plt.show()
# print(theta1)
# print(b)

# c = orig_data[52:53, :5]
# print(c.T)
# d = model(c,theta1)
# print(d)
# print(model(c,theta1))
f=90
g=91
h =10
j = 0

while f < 100:
    c = new_data[f:g, :5]
    d = model(c, theta1)
    i = new_data[f:g, 5:6]
    if (d<0.5 and i<0.5):
        j +=1
    elif (d>=0.5 and i >0.5):
        j+=1
    f += 1
    g += 1
print("取90%时正确率为",j/h)