#导入数据分析的三大件
import  numpy as np
import numpy.random
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #该函数将矩阵随机划分为训练子集和测试子集

import seaborn as sns
from sklearn.linear_model import LogisticRegression
#导入鸢尾花数据集
from sklearn.datasets import load_iris
from sklearn import datasets
#第一步读进去数据看看数据的样子
iris_datas = load_iris()
 #取前两列数据
print(iris_datas.data[:,:3])#数据集中的前3列数据
print(iris_datas.target[0:100])#iris种类,取前两类
iris = pd.DataFrame(iris_datas.data[:,:3],columns=['SpealLength','SpealWidth','PetalLength'])
iris_target = pd.DataFrame(iris_datas.target[0:100],columns=['Types'])
print(iris.shape)
print(iris.head())
print(iris_target.head())
#绘图
X = iris_datas.data[:,:2]
plt.scatter(X[0:50,0],X[0:50,1],s=30,c='b',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],s=30,c='r',marker='x',label='versicolor')
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("trained points", fontsize = 15)
plt.show()
#逻辑回归，目标：建立分类器即决策边界（求解出θ0θ1θ2） 设定阈值，根据阈值判断结果
#sigmoid模块:由值到概率的映射
def sigmoid(z):
    return 1/(1+np.exp(-z)) #np.exp()表示e的多少幂次方
#预测函数模块
def model(x_train,theta):
    return sigmoid(np.dot(x_train,theta.T))#np.dot()表示矩阵的乘法
#加入全1列
iris.insert(0,'Ones',1)
orig_data = iris.as_matrix()
print("打印矩阵")
print(orig_data)

orig_target = iris_target.as_matrix()
cols = orig_data.shape[1]
cols_t = orig_target.shape[0]
x_train = orig_data[0:100,0:cols-1]
y = orig_target[:,0:cols_t-1] #标签矩阵
theta = np.zeros([1,3])
print(x_train[:5])
print(y[:5])
print(theta)
print(x_train.shape,y.shape,theta.shape)




#定义损失函数
def cost(x_train,y,theta):
    left = np.multiply(-y,np.log(model(x_train,theta)))
    right = np.multiply(1-y,np.log(1-model(x_train,theta)))
    return np.sum(left-right)/(len(x_train))
print("初始损失值为：")
print(cost(x_train,y,theta))
#计算梯度
def gradient(x_train,y,theta):
    grad = np.zeros(theta.shape)
    error = (model(x_train,theta)-y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error,x_train[:,j])
        grad[0,j] = np.sum(term)/len(x_train)
    return grad
#选取迭代的停止方案
STOP_ITER = 0 #按指定迭代次数停止
STOP_COST = 1
STOP_GRAD = 2
def stopCriterion(type,value,threshold):
    if type == STOP_ITER:return value>threshold
    elif type == STOP_COST:return abs(value[-1]-value[-2])<threshold
    elif type == STOP_GRAD:return  np.linalg.norm(value)<threshold

#打乱数据--洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    # cols_t = data_t.shape[0]
    x_train = data[0:100,0:cols-1]
    # y = data_t[:,0:cols_t-1]
    return x_train
def shuffleData_t(data_t):
    np.random.shuffle(data_t)
    cols_t = data_t.shape[0]
    y = data_t[:, 0:cols_t - 1]
    return y
#梯度下降求解
#batchSize=1:随机梯度下降；=总样本数：梯度下降；=1-总体：小梯度下降
#stopType：停止策略
#thresh:停止策略对应的阈值
#alpha:学习率
def descent(data,data_t,theta,batchSize,stopType,thresh,alpha):
    int_time = time.time()
    i = 0 #迭代次数初始为0
    k = 0 #batch
    x_train = shuffleData(data)
    y = shuffleData_t(data_t)
    grad = np.zeros(theta.shape) #计算的梯度
    costs = [cost(x_train,y,theta)] #损失值

    while True:
        grad = gradient(x_train[k:k+batchSize],y[k:k+batchSize],theta)
        k += batchSize #取batch数量个数集
        if k>= n:
            k = 0
            x_train = shuffleData(data) #重新洗牌
            y = shuffleData_t(data_t)
            theta = theta - alpha*grad#参数更新
            costs.append(cost(x_train,y,theta))#计算新的损失
            i += 1

            if stopType == STOP_ITER: value = i
            elif stopType == STOP_COST: value = costs
            elif stopType == STOP_GRAD: value = grad
            if stopCriterion(stopType,value,thresh):break

    return theta,i-1,costs,grad,time.time() - int_time
def runExpe(data,data_t,theta,batchSize,stopType,thresh,alpha):
    theta,iter,costs,grad,dur = descent(data,data_t,theta,batchSize,stopType,thresh,alpha)
    name = "Original" if(data[:,1]>2).sum()>1 else "Scaled"
    if batchSize==n :strDescType = "Gradient"
    elif batchSize==1:strDescType = "Stochastic"
    else:strDescType = "Mini-batch({})".format(batchSize)
    name+=strDescType+"descent-Stop:"
    if stopType == STOP_ITER:strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:strStop = "costs change < {}".format(thresh)
    else:strStop = "gradient norm < {}".format(thresh)
    name+=strStop
    print("***{}\nTheta:{}-Iter:{}-Last cost:{:03.2f}-Duration:{:03.2f}s".format(
        name,theta,iter,costs[-1],dur))
    # fig,ax = plt.subplots(figsizw=(12,4))
    plt.plot(np.arange(len(costs)),costs,'r')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title(name.upper()+'Error vs.Iteration')
    plt.show()
    return theta 

n=100
runExpe(orig_data,orig_target,theta,n,STOP_ITER,thresh=5000,alpha=0.000001)

X_train ,X_test , y_train , y_test = train_test_split(orig_data[0:90,0:2],orig_target[0:90,0:2],test_size=0.3,random_state=0)
print(X_train.shape , y_train.shape ,X_test.shape , y_test.shape)

lr = LogisticRegression(penalty='l2',solver='newton-cg',multi_class='multinomial')
lr.fit(X_train,y_train) #fit(x,y)方法来训练模型。x为数据，y为数据所属类型

print("模型在测试集上的分类正确率：%.3f" %lr.score(X_test, y_test))










