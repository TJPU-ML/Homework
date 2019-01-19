import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

y=list();
max_iter=500
alpha=0.01        ##设置步长和迭代次数 
my_testsize=0.1   ##设置测试集大小                                      

def pre_Data(my_testsize):                                                          ##输入测试集的比例：my_testsize
    getdata=pd.read_table(r'irisdata.txt',header=None,sep=',')
    dataset=getdata.replace(['Iris-setosa','Iris-versicolor'],[1,0])                ##从文件读取数据，把两个用字符串表示的类别名替换为1，0
    X=np.array(dataset.ix[:,:3])
    X_add1=np.ones(X.shape[0])
    X=np.c_[X,X_add1]                                                               ##给每个特征向量后面加上1
    y=np.array(dataset[4])                                                          ##分割数据集，得到特征值X和标签y
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=my_testsize,random_state=0,stratify=y)
    return np.mat(X_train),np.mat(X_test),np.mat(y_train),np.mat(y_test)            ##分别返回测试集和训练集的属性集、标签集

def gradient(y_train,X_train,weight):                                               ##计算梯度
    error=y_train-np.exp(X_train*weight)/(1+np.exp(X_train*weight))
    return (-1)*X_train.T*error

def logistic_regression():                                                          ##对率回归
    X_train,X_test,y_train,y_test=pre_Data(my_testsize)
    y_train=y_train.T                                                              
    weight=np.random.rand(5,1)                                                      ##随机在[0,1)范围内生成 W值和 bias值,                                                                     ##设置步长与迭代次数
    for iter in range(max_iter):
        grad=gradient(y_train,X_train,weight)
        weight=weight-alpha*grad
        cost_function(weight,y_train,X_train)
    return weight

def cost_function(weight,y_train,X_train):
    cost_value=-1*y_train.T*X_train*weight+np.log(1+np.exp(X_train*weight))
    value=cost_value[0,0]
    y.append(value)
def accuracy_count():                                                                 ##计算分类正确率
    my_testsize=0.5                     #############################################更正！
    X_train,X_test,y_train,y_test=pre_Data(my_testsize)
    y_test=y_test.T
    weight=logistic_regression()
    y_pred= 1/(1+np.exp(-X_test*weight))                        ##计算预测值
    count=0
    for i in range(y_test.shape[0]-1):
        if np.absolute(y_pred[i]-y_test[i])>=1e-1:
            count=count+1
    return (1-(count/y_test.shape[0]))

def Plot():
    x=np.linspace(1,max_iter,max_iter)
    plt.figure()
    plt.plot(x,y,color='red',linewidth=2)
    plt.xlabel("training_times")
    plt.ylabel("loss function")
    plt.show()
    



