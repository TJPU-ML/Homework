#导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#导入数据
import os
path = '/Users/wangchenchen/PycharmProjects/Test project/iris.data.csv'
data = pd.read_csv(path, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
data.head()

#画散点图
positive = data[data['species'] == 1]#setosa用1代替
negative = data[data['species'] == 0]#virginica用0代替

plt.figure(figsize=(10, 5))  #画图域
plt.scatter(positive['sepal_length'], positive['sepal_width'], s=30, c='blue', marker='o', label='setosa')
plt.scatter(negative['sepal_length'], negative['sepal_width'], s=30, c='red', marker='x', label='virginica')
plt.legend()
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
#plt.show()

#映射到概率的函数
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#返回预测结果值
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


data.insert(0, 'Ones', 1) #插入一列为1的值
orig_data = np.array(data)
cols = orig_data.shape[1]
X = orig_data[:, 0:cols-1]
y = orig_data[:, cols-1:cols]
theta = np.zeros([1, 5])
#计算损失值
def cost(X, y, theta):
    left = np.multiply(y, np.log(model(X, theta)))
    right = np.multiply(1-y, np.log(1 - model(X, theta)))
    return np.sum(left + right) / (-len(X))


#计算参数的梯度方向
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta)-y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)

    return grad


#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

#进行参数更新
import time
def descent(data, theta, batchSize, thresh, alpha):
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #batch
    X,y = shuffleData(data)
    grad = np.zeros(theta.shape) #计算的梯度
    costs = [cost(X, y, theta)]  #损失值
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k = k + batchSize  #去batch数量个数据
        if k >= 100:
            k = 0
            X, y = shuffleData(data)  #重新洗牌
        theta = theta - alpha * grad  #更新参数
        costs.append(cost(X, y, theta))  #计算新的损失
        i = i + 1
        if i > thresh:
            break
    # 损失函数变化图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(costs)), costs, c='red')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(' - Error vs. Iteration')
    plt.show()
    return theta


f=descent(orig_data, theta, 100, 10000, 0.005)




#设定阈值
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]

predictions = predict(X,f)
correct = [1 if ((a == 1 and b == 1)or (a == 0 and b == 0)) else 0 for (a,b) in zip(predictions, y)]
accuracy = (sum(map(int,correct)) / len(correct))
print('accuracy={0}'.format(accuracy))








