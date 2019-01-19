#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
######################################数据导入########################
data_t=[]
data_f=np.loadtxt('E:/biancheng/ml/iris.data',delimiter=",",usecols=(0,1,2,3),dtype=float)
data_s=np.loadtxt('E:/biancheng/ml/iris.data',delimiter=",",usecols=4,dtype=str)
for i in range(0, len(data_s)):
    if data_s[i] =='Iris-setosa':
        data_t.append(1)
    else:
        data_t.append(0)
data=np.insert(data_f,4,values=data_t,axis=1)
x=data[:100,1:3]
y=data[:100,4]
m,n=np.shape(x)
####################################数据显示################
seto=data[:50]
vers=data[51:100]
f1=plt.figure(1)
plt.title('Iris')
plt.xlabel('Width')
plt.ylabel('Length')
plt.scatter(seto[:,0],seto[:,2],marker='o',color='r',s=100,label='seto')
plt.scatter(vers[:,0],vers[:,2],marker='o',color='k',s=100,label='vers')
plt.legend(loc='upper right')
plt.show()
###########################Sigmoid########################################
def sigmoid(x,beta):#(3.18)
    return  1.0/(1+np.math.exp(-np.dot(beta,x)))
####################梯度下降法####################
def Gradientdescent(x,y,theta):
    loop_max=500 #最大迭代次数

    alpha=0.01
    xT=np.transpose(x)
    for i in range(0,loop_max):
        hypothesis=np.dot(x,theta)
        loss=hypothesis-y
        gradient=np.dot(xT,loss)/m
        theta=theta-alpha*gradient
    return theta

################预测函数#####################
def predict (x,theta):
    m,n=np.shape(x)
    xTest=np.ones((m,n+1))
    xTest[:,:-1]=x
    yp=np.dot(xTest,theta)


    return yp
#################损失函数##############
def runExpe(x,y,theta):
    costs=Gradientdescent(x,y,theta)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs,'r')
    plt.show()
    return  costs
#############划分训练集与测试集###########
m,n=np.shape(x)
theta = np.zeros(n)
costs= np.zeros(n)
np.ones(n)
#x_hat=np.c_[x,np.ones(m)]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.5,random_state=0)
theat=Gradientdescent(x_test,y_test,theta)
costs=runExpe(x,y,theat)
y_p=predict(x_test,theat)
print(y_p)
