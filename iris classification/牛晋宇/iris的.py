import numpy as np
import  pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
def shuffleData(data):  #将数据随机打乱
    np.random.shuffle(data)
    cols=data.shape[1]
    X=data[:,0:cols-1]
    y=data[:,cols-1:cols]
    return X,y
def sigmoid(z):         #构造sigmoid函数
    return 1/(1+np.exp(-z))
def model(X,theta):     #模型的建立
    return sigmoid(np.dot(X,theta.T))
def cost(X,y,theta):         #计算
    left=np.multiply(-y,np.log(model(X,theta)))
    right=np.multiply((1-y),np.log(1-model(X,theta)))
    return np.sum(left-right)/len(X)
def gradient(X, y, theta):        #计算函数梯度
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:,j])
        grad[0, j] = np.sum(term) / len(X)
    return grad

def descnet (data,theta,batchSize, alpha):  #进行批量梯度下降法,当costs 值小于等于0。1时，停止迭代
    i=0
    k=0
    X,y=shuffleData(data)
    grad=np.zeros(theta.shape)
    costs=[cost(X,y,theta)]
    while True:
        grad=gradient(X[k:k+batchSize,:],y[k:k+batchSize,:],theta)
        k+=batchSize
        if (k>=batchSize):
            k=0
            X,y=shuffleData(data)
            theta=theta-alpha*grad
            costs.append(cost(X,y,theta))
            i=i+1
            if(costs[i]<=0.1):
                break
    return theta,costs,i
def runExpe(data,theta,batchSize,alpha):    #将随着计算次数的增加，损失函数值打变化图
    theta,costs,i=descnet(data,theta,batchSize,alpha)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs,'r')
    plt.show()
    return theta,costs,i
def predict(X,theta):       #用测试集对所构造的模型进行测试
    return [1 if x>0.5 else 0 for x in model(X,theta)]
iris = load_iris()    #数据集
X_data=pd.DataFrame(iris.data[50:150,:])   # 取数据集第二，三种花的属性
y_data=pd.DataFrame(iris.target[50:150])   # 取数据集第二，三种花的target
data=pd.concat([X_data,y_data],axis=1)     # 将数据属性和target以行来接拼在一个dateFrame
data.insert(0,'bias',1)                    # 创建把数据的第0列名称为'bias'并将这一列所有值赋为1
cols=data.shape[1]                         # 计算数据有多少列
orig_data=np.array(data)                   # 将数据变为以矩阵的形式
orig_data[50:100,5]=1                      # 将第三种花的target定义为1
orig_data[0:50,5]=0                        # 将第二种种花的target定义为0
data_train=np.vstack((orig_data[0:25,:],orig_data[50:75,:]))     #将数据的训练集划分出来
data_exam=np.vstack((orig_data[25:50,:],orig_data[75:100,:]))    #将数据的测试集划分出来
X_train=data_train[:,0:cols-1]
y_train=data_train[:,cols-1:cols]
X_exam=data_exam[:,0:cols-1]
y_exam=data_exam[:,cols-1:cols]
theta=np.zeros([1,5])
batchSize=50
theta,costs,i=runExpe(data_train,theta,batchSize,0.01)
print('MinCosts',costs[i])
print('MIntimes',i)
predictions=predict(X_exam,theta)
correct=[1 if((a==1 and b==1) or (a==0 and b==0)) else 0 for(a,b) in zip(predictions, y_exam)]
accuracy=sum(map(int,correct))/len(correct)
print('Accuracy:',accuracy)



