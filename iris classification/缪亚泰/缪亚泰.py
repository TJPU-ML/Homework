import numpy as np  # 科学计算（矩阵）包
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

# 数据读取
# 取前100行数据进行
iris_path = 'iris.csv'
iris = pd.read_csv(iris_path)


def iris_type(s):
    class_label = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    return class_label[s]


new_iris = pd.read_csv(iris_path, converters={5: iris_type})
new_iris = np.array(new_iris)
# 数据存储，其中：x(m*n矩阵，m为样本个数，n为特征数)，y(m维列向量)
x = new_iris[0:100, 1:-1]
y = new_iris[0:100, -1]

# 以0为种子随机选择50%的数据作为测试集，剩余作为训练集
x = np.mat(x)
y = np.mat(y)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)


# sigmoid函数
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


# 预测函数
def predict(X_train, beta):
    return sigmoid(X_train * beta)


# 梯度下降法
def tdxjf(X_train, Y_train, ddcs, alpha):
    beta = np.ones((X_train.shape[1], Y_train.shape[1]))
    for i in range(ddcs):
        loss = predict(X_train, beta) - Y_train  # 预测值减去实际值
        beta = beta - alpha * X_train.transpose() * loss
    return beta


# 测试
beta = tdxjf(X_train, y_train, 15, 0.001)
test_predict = predict(X_test, beta)
test_predict = np.round(test_predict)

metrics.accuracy_score(y_test, test_predict)
rate = sum(y_test==test_predict) / len(y_test)

print('正确率为', int(rate*100), '%')
