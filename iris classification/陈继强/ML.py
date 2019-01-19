import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.datasets import load_iris
iris=load_iris()      #从sklearn库中导入鸢尾花数据集
X=iris.data[25:75]    #训练集，X有4个属性
y=iris.target[25:75]  #训练集，X的标记，前50个数据为0，后50个数据为1
E=iris.data[75:100]   #测试集，E等同于X
f=iris.target[75:100] #测试集，f等同于y
X=pd.DataFrame(iris.data[25:75])   #转换成表，为插入做准备
y=pd.DataFrame(iris.target[25:75])
E=pd.DataFrame (iris.data[75:100])
f=pd.DataFrame(iris.target[75:100])
X.insert(0,'ones',1)   #在第一列插入一列1
E.insert(0,'ones',1)
temp=pd.concat([X,y],axis= 1)   #将标记与数据连接起来
temp1=pd.concat([E,f],axis=1)
data1=np.array(temp)
X=np.array(X)
E=np.array(E)
f=np.array(f)
y=np.array(y)
w=np.zeros([1,5])  #为w参数初始化0
n=10
def sigmoid(z):
    return 1/(1+np.exp(-z))   #sigmoid函数原型
def model(X,w):
    return sigmoid(np.dot(X,w.T))   #模型函数，线性对数几率回归模型
def cost(X,y,w):
   left=np.multiply(-y,np.log(model(X,w)))
   right=np.multiply(1-y,np.log(1-model(X,w)))   #构造损失函数
   return np.sum(left-right)/(len(X))
def gradient(X,y,w):
    grad=np.zeros(w.shape)    #进行占位
    error=(model(X,w)-y).ravel()
    for j in range(len(w.ravel())):    #对每一个w参数求导
        term=np.multiply(error,X[:,j])
        grad[0,j]=np.sum(term)/len(X)
    return grad
def shuffleData(data):
    np.random.shuffle(data)
    cols=data.shape[1]    #计算列值大小
    X=data[:,0:cols-1]    #增强泛化能力，每当梯度下降时，打乱数据
    y=data[:,cols-1:cols]
    return X,y
def decent(data,w,batchSize,thresh,alpha):
    init_time=time.time()
    i = 0
    k = 0
    X,y=shuffleData(data)
    grad=np.zeros(w.shape )
    costs = [cost(X,y,w)]
    while True:
        grad=gradient(X[k:k+batchSize],y[k:k+batchSize],w)
        k+=batchSize
        if k >= n:
            k=0
            X,y=shuffleData(data)
        w=w-alpha *grad      #进行w参数更新
        costs.append(cost(X,y,w))   #w参数每更新一次，计算损失值
        i+=1
        if(i>thresh ):
            break
    return w,i-1,costs,grad,time.time()-init_time

def runExpe(data,w,batchSize,thresh,alpha):
      w,iter,costs,grad,dur=decent(data,w,batchSize,thresh,alpha )
      print ("w:{}-Iter:{}-last cost:{:03.2f}-Duration:{:03.2f}s".format(w,iter,costs[-1],dur))
      fig,ax=plt.subplots(figsize=(12,4))
      ax.plot(np.arange(len(costs)),costs,'r' )
      ax.set_xlabel('Iterations')
      ax.set_ylabel('cost')                       #对结果的输出，损失函数图像显示
      ax.set_title('Costs with Iteration')
      plt.show()
      return w
e=runExpe(data1,w,n,thresh= 10000,alpha= 0.001)
def predict(X,w):
    return [1 if x>=0.5 else 0 for x in model(X,w) ]    #测试模型，确立阈值0.5，小于0.5判为0，大于0.5判为1
predictions=predict(E,e)
correct=[1 if ((a==1 and b==1)or (a==0 and b==0)) else 0 for(a,b) in zip(predictions,f)]
accuracy=(sum(map(int,correct))/len(correct))
print('accuracy={0}'.format(accuracy) )    #计算准确率




