import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_iris_data(split_per):
    '''
    函数功能：划分数据集
    :param split_per: 数据集之测试集占比
    :return: X_train,X_test,y_train, y_test
    '''
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=split_per, random_state=0)
    return X_train,X_test,y_train, y_test

# 定义sigmoid函数
def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def LogisticRegression(datas,labels,X_test):
    '''
    :param datas:
    :param labels:
    :param X_test:
    :return: probM
    '''
    kinds = list(set(labels))               #3个类别的名字列表
    means = datas.mean(axis=0)              #各个属性的均值
    stds = datas.std(axis=0)                #各个属性的标准差
    N,M =  datas.shape[0],datas.shape[1]+1  #N是样本数，M是参数向量的维
    K = np.shape(list(set(labels)))[0]      #K是类别数

    data = np.ones((N,M))
    data[:,1:] = (datas - means)/stds   #对原始数据进行标准差归一化
    W = np.zeros((K-1,M))               #存储参数矩阵
    priorEs = np.array([1.0/N*np.sum(data[labels == kinds[i]],axis=0) for i in range(K-1)]) #各个属性的先验先验期望值

    loss_delta=[]    #绘图y坐标
    for it in range(1000):
        wx = np.exp(np.dot(W,data.transpose()))
        probs = np.divide(wx,1+np.sum(wx,axis=0).transpose())  #K-1*N的矩阵
        posteriorEs = 1.0/N*np.dot(probs,data)                 #各个属性的后验期望值
        gradients = posteriorEs - priorEs + 1.0/1000 *W         #梯度，最后一项是高斯项，防止过拟合
        W  -= gradients #对参数进行修正
        loss_delta.append(np.sum(posteriorEs - priorEs))

    xx=[i for i in range(1000)]
    fig, ax = plt.subplots()
    ax.plot(xx, loss_delta, 'b-')
    ax.set_xlabel('item number')
    ax.set_ylabel('loss change')
    ax.set_title("loss of line")
    plt.show()

    #probM每行三个元素，分别表示data中对应样本被判给三个类别的概率
    means1 = X_test.mean(axis=0)  # 各个属性的均值
    stds1 = X_test.std(axis=0)  # 各个属性的标准差
    N1, M1 = X_test.shape[0], X_test.shape[1] + 1  # N是样本数，M是参数向量的维
    data1 = np.ones((N1, M1))
    data1[:, 1:] = (X_test - means1) / stds1  # 对原始数据进行标准差归一化
    probM = np.ones((N1,K))
    probM[:,:-1] = np.exp(np.dot(data1,W.transpose()))
    probM /= np.array([np.sum(probM,axis = 1)]).transpose() #得到概率
    #print("当前权值:",W)
    return probM

def predict(probM):
    '''概率转换为0 1 2标签'''
    y_pre=[]
    for line in probM:
        line=list(line)
        y_pre.append(line.index(max(line)))
    return y_pre

if __name__ == "__main__":
    split_list=[0.1,0.3,0.5]
    for i in split_list:
        X_train, X_test, y_train, y_test=load_iris_data(i)
        probM=LogisticRegression(X_train,y_train,X_test)
        y_pre=predict(probM)
        print("测试占比:{} 准确率:{}".format(i,accuracy_score(y_pre,y_test)))
    iris = load_iris()
    DD = iris.data
    X1 = [x[0] for x in DD]
    Y1 = [x[1] for x in DD]
    # plt.scatter(X, Y, c=iris.target, marker='x')
    plt.scatter(X1[:50], Y1[:50], color='red', marker='*', label='0')  # 前50个样本
    plt.scatter(X1[50:100], Y1[50:100], color='blue', marker='o', label='1')  # 中间50个
    plt.scatter(X1[100:], Y1[100:], color='green', marker='x', label='2')  # 后50个样本
    plt.legend(loc=2)  # 左上角
    plt.show()