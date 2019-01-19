import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import numpy.random
import time
import imp

#from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x_data = pd.DataFrame(iris.data[1:100])
y_data = pd.DataFrame(iris.target[1:100])
data = pd.concat([x_data, y_data], axis=1)

data.insert(0, 'bias', 1)
cols = data.shape[1]
data1 = np.array(data)

data_train = np.vstack((data1[1:25, :],data1[76:100, :])) #人工将训练集和测试集分开
data_exam = np.vstack((data1[26:50, :],data1[51:75, :] ))
x_train = data_train[:, 0:cols-1]
y_train = data_train[:, cols-1:cols]
x_exam = data_exam[:, 0:cols-1]
y_exam = data_exam[:, cols-1:cols]
theta = np.zeros([1, 5])

#classifier = KNeighborsClassifier()

# sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

# 插入一列值都为1的数，把数值运算转变为矩阵运算
def model(x,theta):
    return sigmoid(np.dot(x, theta.T))

#损失函数
def cost( x, y, theta ) :
   left = np.multiply(-y, np.log(model(x, theta)))
   right = np.multiply(1-y, np.log(1-model(x, theta)))
   return np.sum(left-right)/(len(x))

#梯度计算
def gradient ( x , y , theta):
    grad = np.zeros(theta.shape)    #进行占位
    error = (model(x, theta)-y).ravel()
    for j in range(len(theta.ravel())):    #对每一个theta参数求导
        term = np.multiply(error, x[:, j]) #取第j列
        grad[0, j] = np.sum(term)/len(x)
    return grad

#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    x = data[:, 0:cols-1]
    y = data[:, cols-1:cols]
    return x, y

#梯度下降求解
n = 50#梯度
def decent(data, theta, batchSize, thresh, alpha):
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #batch
    x, y = shuffleData(data)#再洗牌
    grad = np.zeros(theta.shape )#参数更新，计算梯度
    costs = [cost(x, y, theta)] #计算损失

    while True:
        grad = gradient(x[k:k+batchSize], y[k:k+batchSize], theta)
        k+=batchSize
        if k >= n:
            k = 0
            x, y =shuffleData(data)#重新洗牌
        theta = theta-alpha*grad #参数更新
        costs.append(cost(x, y, theta))#计算新的损失
        i+= 1
        if(i > thresh ):
            break
    return theta, i-1, costs, grad, time.time()-init_time

#精度，设定阈值（概率值改变成类别值）
def predict(x, theta):
    return [1 if x > 0.5 else 0 for x in model(x, theta)]


#画图
def runExpe(data, theta, batchSize, thresh, alpha):
    theta, iter, costs, grad, dur = decent(data, theta, batchSize, thresh, alpha)
    print("theta:{}-Iter:{}-last cost:{:03.2f}-Duration:{:03.2f}s".format(theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Costs with Iteration')
    plt.show()
    return theta, costs

batchSize=20
theta,costs=runExpe(data_train, theta, batchSize, 100000, 0.0001)
#print('MinCosts',costs[100000])
predictions=predict(x_exam,theta)
correct=[1 if((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for(a, b) in zip(predictions, y_exam)]
accuracy=sum(map(int, correct))/len(correct)
print('Accuracy:', accuracy) #计算准确率
