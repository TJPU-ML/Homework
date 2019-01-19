import numpy as np
import random

from numpy import genfromtxt,zeros
data=genfromtxt('iris.data.txt',delimiter=',',usecols=(0,1,2,3))
target=genfromtxt('iris.data.txt',delimiter=',',usecols=(4),dtype=str)
t=zeros(len(target))
t[target=='Iris-setosa']=0
t[target=='Iris-versicolor']=1


#将数据集按比例随机分为训练集和测试集

def trainTestSplit(trainingSet,trainingLabels,test_size):
    totalNum=int(len(trainingSet))
    trainIndex=list(range(totalNum))#存放训练集的下标

    testIndex=[]  #测试集的下标
    x_test=[]     #测试集输入
    y_test=[]     #测试集输出
    x_train=[]    #训练集输入
    y_train=[]    #训练集输出

    trainNum = int(totalNum * test_size) #划分训练集的样本数

    for i in range(trainNum):
        randomIndex = int(random.uniform(0,len(trainIndex)))
        x_test.append(trainingSet[randomIndex])
        y_test.append(trainingLabels[randomIndex])
        del(trainIndex[randomIndex])#删除已经放入测试集的下标

    for i in range(totalNum-trainNum):
        x_train.append(trainingSet[trainIndex[i]])
        y_train.append(trainingLabels[trainIndex[i]])

    return x_train,y_train,x_test,y_test

#比例
all_data=trainTestSplit(data,t,0.1)
x_train=all_data[0]
y_train=all_data[1]

x_test=all_data[2]
y_test=all_data[3]


#logistic函数
def logistic(x):
    return 1.0/(1+np.exp(-x))

#模型
def model(X_train,w):
    return logistic(X_train*w)


###梯度下降，求w
def gradAscent(dataIn,Target):
    X=np.mat(dataIn) 
    Y=np.mat(Target).transpose()

    m,n=np.shape(X)   #矩阵的行列
    alpha=0.001    #步长
    maxCycle=1000  #迭代次数

    weights=np.ones((n,1))   #n*1的数组

    for k in range(maxCycle):
        y_hat=model(X,weights)

        #print('y_hat: ',y_hat)

        error=(y_hat-Y)
        weights=weights-alpha*X.transpose()*error

    #print(y_hat)
    return weights

W=gradAscent(x_train,y_train)
#print (W)
y_test_hat=model(x_test,W)
print (y_test_hat)

#正确率
def correct(y_Test_hat,y_Test):
    y_Test_hat[y_Test_hat<=0.5]=0
    y_Test_hat[y_Test_hat>=0.5]=1
    n=len(y_Test_hat)
    y=y_Test_hat.reshape((-1,n))
    #print (y)
    result_error=y-y_Test
    #print (result_error)
    wrong_num=result_error[result_error!=0]
    error_sum=len(wrong_num)
    correct=(n-error_sum)/n
    print (correct)
    return correct

correct(y_test_hat,y_test)

    



    

