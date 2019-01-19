import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义回归模型
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


# 计算梯度
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)  # 初始化梯度，维度与参数向量的维度相同
    error = (model(X, theta) - y).ravel()  # 计算偏差
    for j in range(len(theta.ravel())):  # 计算n个偏导数（梯度）
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)
    return grad


# 定义损失函数
def cost(X, y, theta):
    return np.sum((np.multiply(-y, np.log(model(X, theta)))) - (np.multiply(1 - y, np.log(1 - model(X, theta)))))/(len(X))

path = 'datas' + os.sep + 'iris.csv'
irisData = pd.read_csv(path, header=None, names=['petal_len', 'petal_width', 'sepal_len', 'sepal_width', 'class'],
                       dtype={'petal_len': float, 'petal_width': float, 'sepal_len': float, 'sepal_width': float,
                              'class': str})
irisData.loc[irisData['class'] == 'setosa', 'class'] = 0  # 将setosa置为0
irisData.loc[irisData['class'] == 'versicolor', 'class'] = 1  # 将versicolor置为1
irisData.loc[irisData['class'] == 'virginica', 'class'] = 2  # 将virginica置为2

print("---------------打印数据信息------------------ #")
print(irisData.head())  # 打印前两行
print(irisData.shape)  # 打印数据维度
print(irisData.describe())  # 打印描述信息
print()

# 绘制数据分布图像
positive = irisData[irisData['class'] == 0]  # 设置正类
negative = irisData[irisData['class'] == 1]  # 设置负类

# fig, ax = plt.subplots(figsize=(8, 6))
fig, figer1 = plt.subplots(figsize=(10, 5))  # 设置图像大小
figer1.scatter(positive['sepal_len'], positive['sepal_width'], s=30, c='b', marker='o', label='setosa')  # 绘制setosa花的散点图
figer1.scatter(negative['sepal_len'], negative['sepal_width'], s=30, c='r', marker='x',
               label='versicolor')  # 绘制versicolor花的散点图
figer1.legend(loc=2)  # 标题放在左上角
figer1.set_xlabel('sepal_len')  # 设置x标签
figer1.set_ylabel('sepal_width')  # 设置y标签
plt.show()  # 显示初始图像

irisData.insert(2, 'Ones', 1)  # 在第3列插入一列数据,值为1

print("----------打印初始数据的前五行------------ ")
print(irisData.head())

orig_data = irisData.as_matrix()  # 构造一个矩阵
print(orig_data.dtype)
print("----------------初始打印矩阵-----------------")
print(orig_data[:5, :])

cols = orig_data.shape[1]  # 得到矩阵的列数
orig_data = orig_data[:100, :]  # 取矩阵的前100行数据

scaled_data1 = orig_data[:50, 2:cols]  # 第一类数据矩阵，选择花瓣属性
scaled_data2 = orig_data[50:100, 2:cols]  # 第二类数据矩阵
np.random.shuffle(scaled_data1)  # 打乱第一类数据的顺序
np.random.shuffle(scaled_data2)  # 打乱第二类数据的顺序
np.random.shuffle(orig_data)
# 从两个矩阵中分别取固定个数的数据作为测试集
# scaled_data = orig_data[4:100, 2:cols]
#scaled_data = np.vstack((scaled_data1[:25, :], scaled_data2[:25, :]))  # 50%
#scaled_data = np.vstack((scaled_data1[:15, :], scaled_data2[:15, :]))  # 30%
scaled_data = np.vstack((scaled_data1[:5, :], scaled_data2[:5, :]))  # 10%


np.random.shuffle(scaled_data)  # 打乱测试集数据的顺序

print("-------打印测试集-------")
print(scaled_data)
print("------测试集的属性-------")
print(scaled_data.shape)

# 从两个矩阵中分别取相同个数的数据作为训练集
# orig_data = orig_data[:4, 2:cols]
#orig_data = np.vstack((scaled_data1[25:50, :], scaled_data2[25:50, :]))  # 50%
#orig_data = np.vstack((scaled_data1[15:50, :], scaled_data2[15:50, :]))  # 70%
orig_data = np.vstack((scaled_data1[5:50, :], scaled_data2[5:50, :]))   # 90%

np.random.shuffle(orig_data)  # 打乱训练集数据的顺序

print("---------打印训练集--------")
print(orig_data)

X = orig_data[:100, 1:cols - 1]  # 选择前三列
y = orig_data[:100, cols - 1:cols]  # 选择最后一列结果
print("-------打印X的值-------")
print(X)
print("---------打印y的值----------")
print(y)

# 构造参数向量
theta = np.zeros([1, 3])

# 打印矩阵的维度
print("----------打印训练数据的信息----------")
print("参数值为：")
print(theta)
print("X的维度为：")
print(X.shape)
print("y的维度为")
print(y.shape)
print("参数的维度为")
print(theta.shape)

c = cost(X, y, theta)  # 求初始损失函数的值
print("--------初始损失值为-------")
print(X.dtype)
print(c)


#  刷新数据,打乱数据的顺序
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:100, 0:cols - 1]
    y = data[:100, cols - 1:]
    return X, y


import time


# 定义梯度下降求解函数
def descent(data, theta, batchSize, threshold, alpha):
    init_time = time.time()  # 设置初始时间
    i = 0  # 设置迭代次数
    k = 0  # batch
    X, y = shuffleData(data)  # 打乱数据
    grad = np.zeros(theta.shape)  # 计算初始的梯度
    costs = [cost(X, y, theta)]  # 计算初始损失函数值

    # 开始迭代
    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)  # 求解梯度值
        k += batchSize  # 取batch个数据
        if k >= n:  # 如果数据取完
            k = 0
            X, y = shuffleData(data)  # 对数据进行重新洗牌
        theta = theta - alpha * grad  # 对参数进行更新
        print(theta)
        cost_new = cost(X, y, theta)  # 计算新的损失值
        print(cost_new)
        costs.append(cost_new)  # 将新的损失之追加到列表末尾
        i += 1  # 更新循环变量

        value = costs  # cost为损失值
        if abs(value[-1] - value[-2]) < threshold:
            break
    return theta, i - 1, costs, grad, time.time() - init_time


# 绘制图像
def Run(data, theta, batchSize, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, thresh, alpha)  # 开始执行梯度下降
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} -".format(alpha)
    # 选择梯度下降策略和停止方案
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch({})".format(batchSize)
    name += strDescType + " descent - stop: "
    strStop = "costs change < {}".format(thresh)
    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(name, theta, iter, costs[-1],
                                                                                           dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name)
    plt.show()
    return theta


# 开始训练模型
n = 100  # 一次读入100个数据进行训练
print("打印矩阵")
print(orig_data)

theta = Run(orig_data, theta, n, thresh=0.000001, alpha=0.1)  # 两次迭代损失函数变化非常小时停止(1e-6)


# 对结果进行测试
# 设定阈值 大于0.5则为1，小于0.5为0
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]


scaled_X = scaled_data[:, :3]  # 设置测试集输入
y = scaled_data[:, 3]  # 正确值

print("--------打印测试的数据---------")
print(scaled_X)

print("----------theta的值为-----------")
print(theta)
predictions = predict(scaled_X, theta)
print("-----------打印预测值-----------")
print(predictions)
print("-------------打印真实值-----------")
print(y)

correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) / len(correct)) * 100
print('正确率 = {0}%'.format(accuracy))


# 设置分割曲线函数
def y1(x2, theta):
    # y = theta[0] + theta[1]* x1 + theta[2] * x2
    x1 = (-(theta[0, 0] + theta[0, 2] * x2)) / theta[0, 1]
    return x1


x2 = np.linspace(0, 5, 1000)
x1 = y1(x2, theta)

fig, figer1 = plt.subplots(figsize=(10, 5))  # 设置图像大小
figer1.scatter(positive['sepal_len'], positive['sepal_width'], s=30, c='b', marker='o', label='setosa')  # 绘制setosa花的散点图
figer1.scatter(negative['sepal_len'], negative['sepal_width'], s=30, c='r', marker='x',
               label='versicolor')  # 绘制versicolor花的散点图
figer1.legend(loc=2)  # 标题放在左上角
figer1.set_xlabel('sepal_len')  # 设置x标签
figer1.set_ylabel('sepal_width')  # 设置y标签
plt.plot(x1, x2, 'r-', linewidth=1)
plt.show()  # 显示结果图像
