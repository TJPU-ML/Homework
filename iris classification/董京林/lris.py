
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def std_data(X_data):
    means = X_data.mean(axis=0)
    stds = X_data.std(axis=0)
    N_=X_data.shape[0]
    M_ = X_data.shape[1] + 1
    X_std = np.ones((N_, M_))
    X_std[:, 1:] = (X_data - means) / stds
    return X_std

def accuracy(y_pre,y_test):
    count=0
    for i,j in zip(y_pre,y_test):
        if i==j:
            count+=1
    return count/len(y_test)

def gradAscent(X_train,y_train,K_num):
    ks = list(set(y_train))
    N=X_train.shape[0]
    M = X_train.shape[1] + 1
    data_train = std_data(X_train)
    Weight = np.zeros((K_num - 1, M))
    temp=[1.0 / N * np.sum(data_train[y_train == ks[i]], axis=0) for i in range(K_num - 1)]
    priEs = np.array(temp)

    loss_change=[]
    for i in range(3000):
        x = np.exp(np.dot(Weight, data_train.transpose()))
        probs = np.divide(x, 1 + np.sum(x, axis=0).transpose())
        pEs = 1.0 / N * np.dot(probs, data_train)
        loss_change.append(np.sum(pEs-priEs))
        gradient = pEs - priEs + 1.0 / 1000 * Weight
        Weight = Weight - gradient

    return Weight,loss_change



if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    percent_list = [0.1, 0.3, 0.5]
    xx = [i for i in range(3000)]
    for i in percent_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
        K_ = len(list(set(y_train)))  # Àà±ðÊý
        W, loss = gradAscent(X_train, y_train, K_)
        plt.figure()
        plt.plot(xx, lost)
        plt.title('loss')
        plt.xlabel('')
        plt.ylabel('')
        plt.show()
        Num = X_test.shape[0]
        data = std_data(X_test)
        prob = np.ones((Num, K_))
        temp = np.dot(data, W.transpose())
        prob[:, :-1] = np.exp(temp)
        prob = prob / np.array([np.sum(prob, axis=1)]).transpose()
        y_pred = []
        for p in Pro:
            P = list(p)
            y_pred.append(P.index(max(P)))
        print(f"{i}:{round(accuracy(y_pred, y_test),2)}")