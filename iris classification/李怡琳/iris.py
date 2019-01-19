import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()                 #导入iris数据集
x=pd.DataFrame(iris.data[:100])    #将数据转换成表
y=pd.DataFrame(iris.target[:100])  #将标签转换成表
x.insert(0,'ones',1)               #在第一列添加一列1（theta0）
theta=np.zeros([1,5])              #创建五个初始值为0的theta参数
a=np.hstack((x,y))                 #x与y沿水平方向叠堆成a
data_train,data_test=train_test_split(a,test_size=0.5)  #以一定的比例建立训练集和测试集
cols=a.shape[1]                    #取a的列数
x_test=data_test[:,0:cols-1]
y_test=data_test[:,cols-1:cols]
def sigmoid(z):                    #对数几率函数
    return 1/(1+np.exp(-z))
def model(x,theta):                #预测函数
    return sigmoid(np.dot(x,theta.T))
def cost(x,y,theta):                #损失函数
    left=np.multiply(-y,np.log(model(x,theta)))
    right=np.multiply(1-y,np.log(1-model(x,theta)))
    return np.sum(left-right)/(len(x))   #转换为梯度下降问题
def gradient(x,y,theta):            #梯度
    grad=np.zeros(theta.shape)
    error=(model(x,theta)-y).ravel()
    for j in range(len(theta.ravel())):  #对每个theta进行计算
        term=np.multiply(error,x[:,j])
        grad[0,j]=np.sum(term)/len(x)
    return grad
def shuffleData(a):           #打乱数据
    np.random.shuffle(a)
    cols=a.shape[1]
    x=a[:,0:cols-1]
    y=a[:,cols-1:]
    return x,y
def descent(a,theta,batchsize,thresh,alpha):  #梯度下降求解
    init_time=time.time()
    i=0                       #迭代次数
    k=0
    x,y=shuffleData(a)        #打乱数据
    grad=np.zeros(theta.shape)#计算的梯度
    costs=[cost(x,y,theta)]   #损失值

    while True:
        grad=gradient(x[k:k+batchsize],y[k:k+batchsize],theta)
        k+=batchsize          #取新的数据
        if (k>=batchsize):
            k=0
            x,y=shuffleData(a)
        theta=theta-alpha*grad         #参数更新
        costs.append(cost(x,y,theta))  #计算新的损失
        i+=1
        if (i>thresh):break
    return theta,costs
def runExpe(a,theta,batchsize,thresh,alpha):
    theta,costs=descent(a, theta, batchsize, thresh, alpha)
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)),costs,'r')   #显示损失值与迭代次数的变化图像
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    plt.show()
    return theta,costs
def predict(x,theta):   #模型测试，确立阈值0.5，大于则为1，小于则为0
    return [1 if X>=0.5 else 0 for X in model(x,theta)]
batchsize=1
theta, costs = runExpe(data_train, theta, batchsize, thresh=100, alpha=0.001)
predictions=predict(x_test,theta)
print('MinCosts',costs[100])
correct=[1 if((a==1 and b==1) or (a==0 and b==0)) else 0 for(a,b) in zip(predictions, y_test)]
accuracy=sum(map(int,correct))/len(correct)
print('Accuracy:',accuracy)  #输出准确率