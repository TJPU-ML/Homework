#-*- encoding:utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 定义sigmoid函数
def sigmoid(X):
    '''线性化为非线性化'''
    return 1.0/(1+np.exp(-X))

def std_data(X_data):
    '''数据std归一化 (x-u/std) u为均值、std是标准差'''
    means = X_data.mean(axis=0)
    stds = X_data.std(axis=0)
    N_=X_data.shape[0]
    M_ = X_data.shape[1] + 1
    X_std = np.ones((N_, M_))
    X_std[:, 1:] = (X_data - means) / stds
    return X_std

def accuracy(y_pre,y_test):
    '''求准确率'''
    count=0
    for i,j in zip(y_pre,y_test):
        if i==j:
            count+=1
    return count/len(y_test)

def predict(Pw):
    '''概率转换为预测值'''
    y_pred=[]
    for p in Pw:
        P=list(p)
        y_pred.append(P.index(max(P)))
    return y_pred

def gradAscent(X_train,y_train,K_num):
    '''
    功能：梯度下降算法 求解权值
    传入参数：X_train,y_train,K_num
    返回：Weight
    '''
    ks = list(set(y_train))
    N=X_train.shape[0]
    M = X_train.shape[1] + 1
    data_train = std_data(X_train)
    Weight = np.zeros((K_num - 1, M))
    temp=[1.0 / N * np.sum(data_train[y_train == ks[i]], axis=0) for i in range(K_num - 1)]
    priEs = np.array(temp)

    loss_change=[]
    for i in range(1500):
        '''训练过程'''
        x = np.exp(np.dot(Weight, data_train.transpose()))
        probs = np.divide(x, 1 + np.sum(x, axis=0).transpose())
        pEs = 1.0 / N * np.dot(probs, data_train)
        loss_change.append(np.sum(pEs-priEs))
        gradient = pEs - priEs + 1.0 / 1000 * Weight  # 梯度，最后一项是防止过拟合
        Weight = Weight - gradient

    return Weight,loss_change

def LogisticRegression(Weight,K,X_test):
    '''
    :param Weight  K  X_test
    :return: prob--概率
    '''
    Num= X_test.shape[0]
    data=std_data(X_test)
    prob = np.ones((Num,K))
    temp=np.dot(data,Weight.transpose())
    prob[:,:-1] = np.exp(temp)
    prob =prob/ np.array([np.sum(prob,axis = 1)]).transpose()  #得到概率
    return prob

def plot_loss(x,y):
    plt.figure()
    plt.plot(x, y)
    plt.title('loss')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

def main(X,y):
    '''
    :param X:
    :param y:
    :return:
    '''
    percent_list = [0.1, 0.3, 0.5]
    xx=[i for i in range(1500)]
    for i in percent_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
        K_ = len(list(set(y_train)))  # 类别数
        W,loss = gradAscent(X_train, y_train, K_)
        plot_loss(xx,loss)
        pro = LogisticRegression(W, K_, X_test)
        y_pred = predict(pro)
        print(f"测试占比:{i}--准确率:{round(accuracy(y_pred, y_test),2)}")

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    main(X,y)