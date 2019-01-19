import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
def sigmoid(inX):# 定义sigmoid函数
    return 1.0/(1+np.exp(-inX))
def std_data(X):
    means = X.mean(axis=0) #均值
    stds = X.std(axis=0) #标准差
    A=X.shape[0] #样本个数
    B= X.shape[1] + 1 #参数维度
    X_std = np.ones((A, B))
    X_std[:, 1:] = (X - means) / stds
    return X_std

def predict(Pw):    #准确率
    y_pred=[]
    for p in Pw:
        P=list(p)
        y_pred.append(P.index(max(P)))
    return y_pred

def gradAscent(X_train,y_train,K_num):#梯度下降法解权值
    loss=[]
    ks = list(set(y_train))
    N=X_train.shape[0]  # N样本数，
    M = X_train.shape[1] + 1  #M参数向量的维
    data = std_data(X_train)
    Weight = np.zeros((K_num - 1, M))  # 存储参数矩阵
    temp=[1.0 / N * np.sum(data[y_train == ks[i]], axis=0) for i in range(K_num - 1)]
    priEs = np.array(temp)  # 期望值

    for i in range(1000):
        wx = np.exp(np.dot(Weight, data.transpose()))
        probs = np.divide(wx, 1 + np.sum(wx, axis=0).transpose())
        pEs = 1.0 / N * np.dot(probs, data)
        loss.append(np.sum(pEs-priEs))
        gradient = pEs - priEs + 1.0 /100 * Weight  # 梯度
        Weight = Weight - gradient  # 修正参数

    plt.figure()
    x=[i for i in range(1000)]
    plt.plot(x,loss)
    plt.title('loss line')
    plt.xlabel('number')
    plt.ylabel('loss')
    plt.show()

    return Weight

def LogisticRegression(Weight,K,X_test):
    N1= X_test.shape[0]
    data=std_data(X_test)
    prob = np.ones((N1,K))
    prob[:,:-1] = np.exp(np.dot(data,Weight.transpose()))
    prob =prob/ np.array([np.sum(prob,axis = 1)]).transpose() #概率
    return prob

def main():
    split_list = [0.1, 0.3, 0.5]# 载入数据
    for i in split_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
        K_num = np.shape(list(set(y_train)))[0]
        W = gradAscent(X_train, y_train, K_num)
        prob = LogisticRegression(W, K_num, X_test)
        y_pre = predict(prob)
        print("测试集:{} 准确率:{}".format(i, accuracy_score(y_pre, y_test)))

if __name__ == "__main__":
    main()