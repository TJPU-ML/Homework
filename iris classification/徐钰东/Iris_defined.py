#引入所需要的第三方库
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
#引入所需的数据
iris_Info=pd.read_csv("iris.csv")
#利用其中两个属性画出散点图
f1 = iris_Info[iris_Info['kind'] == 1]
f2= iris_Info[iris_Info['kind'] == 0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(f1['sepal length'], f1['sepal width'], s=30, c='b', marker='o', label='setosa')
ax.scatter(f2['sepal length'], f2['sepal width'], s=30, c='r', marker='x', label='versicolor')
ax.legend()
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
plt.show()
#添加了一列值全为1的数据，目的是为了填加theta0
b=np.ones(100)
iris_data=np.asmatrix(iris_Info) #将数据转换为二维数组形式
iris_data=np.insert(iris_data, 0, values=b, axis=1)  #插入全为1的一列
random.shuffle(iris_data)                            #随机打乱原始数据
iris_data_train=iris_data[:50,:]
iris_data_test=iris_data[50:100,:]
cols=iris_data.shape[1]
x=iris_data_train[:, 0:5]
y=iris_data_train[:, cols-1:cols]
x_t=iris_data_test[:, 0:5]
y_t=iris_data_test[:, cols-1:cols]                   #将训练集和测试集按比例分开，并将属性列和类别类分别表示出来
#print(x,y)
theta=np.zeros([1,5])

#对数几率函数，将数值转换到[0,1]区间上
def Logistic(z):
    return 1/(1+np.exp(-z))
#得出预估值的模块
def Prediction(x,theta):
    return Logistic(np.dot(x, theta.T))

#求损失函数
def cost(x ,y,theta):
    p1=np.multiply(y, np.log(Prediction(x, theta)))
    p2=np.multiply(1-y, np.log(1-Prediction(x, theta)))
    return np.sum(p1+p2)/(-len(x))


#计算梯度
def gradient(x,y,theta):
    grad=np.zeros(theta.shape)
    error=(Prediction(x,theta)-y)
    for i in range(len(theta.ravel())):
        t=np.multiply(error,x[:,i])
        grad[0,i]=np.sum(t)/len(x)
    return grad

print(gradient(x,y,theta))
#利用全部数据进行梯度下降
def Descent(data,theta,batchsize,threshhold,alpha):
    i=0 #迭代次数
    k=0
    grad = np.zeros(theta.shape)
    costs=[cost(x,y,theta)]
    while True:
        grad=gradient(x[k:k+batchsize], y[k:k+batchsize], theta)
        k+=batchsize
        if k>=100:
            k=0
        theta= theta-alpha*grad         #实现参数的更新
        costs.append(cost(x, y, theta))      #计算损失值
        i= i+1                         #迭代次数计数器
        if i>threshhold:               #达到迭代次数后结束训练
            break
    return theta,i-1,costs,grad
#画出损失函数的图像
def Run(data,theta,batchsize,threshhold,alpha):
    theta,iter,costs,grad=Descent(data,theta, batchsize, threshhold, alpha)         #调用梯度下降函数
    name="data-learning rate:-{}".format(alpha)
    name+="  descent-stop:-{}".format(threshhold)
    name+="iterations"
    fig,ax=plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(costs)), costs, c='red')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper())
    plt.show()
    return theta


th=Run(iris_data_train,theta,100,10000,0.001)   #函数调用
#print(th)
#测试集验证的部分
def Test(x,theta):
    s=Prediction(x,theta)      #根据预测模型得到的数值，认为其大于等于0.5即为1，小于0.5即为0
    if s>=0.5:
       return 1
    else:
       return 0
correct=0   #测试集中预测正确的个数
i=0
j=0
ar=[0 for i in range(50)]
sum=50
for i in range(50):
   ar[i]=Test(x_t[i],th)   #预测值的存储
for j in range(50):
    if ar[j]==y_t[j]:
        correct+=1         #正确值的计数
accuracy=correct/sum
print('%.2f%%' % (accuracy * 100))   #得到模型的分类准确率，以百分数输出

