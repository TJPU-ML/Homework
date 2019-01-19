import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy.random
import time
#对数几率函数
def main (n):
    path = r'iris.datanew.csv'
    pdData = pd.read_csv(path, header=None, names=['s1', 's2', 's3', 's4', 'ss'])
    Blueone = pdData[pdData['ss'] == 1]
    Redone = pdData[pdData['ss'] == 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(Blueone['s1'], Blueone['s2'], s=40, c='b', marker='o', label='flower1')
    ax.scatter(Redone['s1'], Redone['s2'], s=40, c='r', marker='x', label='flower2')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    plt.show()
    pdData.insert(0, 'b', 1)  # （1，X1，X2） 加一截距，更好的表达
    data = pdData.as_matrix()
    clos = data.shape[1]
    X = data[:, 0:clos - 1]
    y = data[:, clos - 1:clos]
    theta = np.zeros([1, 5])
    if n==1:
       # data1 = data[0:24, :]+data[75:100,:]
       #输入n为1，2，3采取不同比例的训练集和测试集
        data1=np.concatenate([data[0:25, :], data[75:100,:]], axis=0)
        data2=np.concatenate([data[25:50, :], data[50:75,:]], axis=0)
    elif n==2:
        data1 = np.concatenate([data[0:35, :], data[50:85, :]], axis=0)
        data2 = np.concatenate([data[35:50, :], data[85:100, :]], axis=0)
    else :
        data1 = np.concatenate([data[0:45, :], data[50:95, :]], axis=0)
        data2 = np.concatenate([data[45:50, :], data[95:100, :]], axis=0)

    theta_new,cost=descent(data1,theta,7000,0.004)
    print(theta_new)

    c=data2.shape[1]
    X=data2[:,0:c-1]
    X2=data2[:,c-1:c]
    i=0
    all=0
    count=0
    for X1 in X:
        all+=1
        one_dim_vec_one = np.array(X1)
        one_dim_vec_two = np.array(theta)
        a=np.dot(X1, theta_new.T)
        b=sigmoid(a)
        print(b)
        if b>=0.5:    #累加正确的个数
            if X2[all-1]==1:
               count+=1
        else:
            if X2[all-1]==0:
                count+=1
    result=count/all
    print(count)
    print(all)

    print("正确率：")
    print(result)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def model(X,theta):
    return sigmoid(np.dot(X,theta.T))
#对数损失函数
def cost(X,y,theta):
    left=np.multiply(-y,np.log(model(X,theta)))
    right=np.multiply(1-y,np.log(1-model(X,theta)))
    return np.sum(left-right)/(len(X))
#梯度
def gradient(X,y,theta):
    grad=np.zeros(theta.shape)
    error=(model(X,theta)-y).ravel()
    for j in range(len(theta.ravel())):
        term=np.multiply(error,X[:,j])
        grad[0,j]=np.sum(term)/len(X)
    return grad

def shuffleData(data):
    np.random.shuffle(data)
    cols=data.shape[1]
    X=data[:,0:cols-1]
    y=data[:,cols-1:]
    return X,y

def descent(data, theta, thresh, alpha):
    init_time = time.time()
    i = 0  # 迭代次数
    X, y = shuffleData(data)
    batchSize=data.shape[0]
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值
    while True:
       # print(theta)
        grad = gradient(X[0: batchSize], y[0:batchSize], theta)  # 取batch数量个数据
        X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1
        if i>thresh: break

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(' 图2')
    plt.show()
    return theta,  costs
#descent(D,theta,70000,0.004)

n=2
main(n);