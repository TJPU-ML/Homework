import numpy as np  # for matrix calculation
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#导入数据
iris_b = np.loadtxt('iris.csv', delimiter=",")
#显示数据
print(iris_b)

X = iris_b[:, 1:3]
y = iris_b[:, 4]
goodData = iris_b[0:50]
badData = iris_b[50:]
#返回数据集大小
m, n = np.shape(X)
print(m, n)

f1 = plt.figure(1)
plt.title('iris_data')
plt.xlabel('Sepal.Width')
plt.ylabel('Petal.Length')
plt.scatter(goodData[:, 1], goodData[:, 2], marker='o', color='g', s=10, label='setosa')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='r', s=10, label='versicolor')
plt.legend(loc='upper right')
plt.show() #显示散点图

#似然函数
def likelihood_sub(x, y, theta):

    return -y * np.dot(theta, x.T) + np.math.log(1 + np.math.exp(np.dot(theta, x.T)))

def likelihood(X, y, theta):

    sum = 0
    m, n = np.shape(X)
    for i in range(m):
        sum += likelihood_sub(X[i], y[i], theta)
    return sum

#激活函数
def sigmoid(x, theta):
    return 1.0 / (1 + np.math.exp(- np.dot(theta, x)))

#梯度下降
def gradDscent_1(X, y):  # implementation of fundational gradDscent algorithms

    h = 0.1  # 迭代步长
    max_times = 500 # 迭代次数的限制
    m, n = np.shape(X)
    #b = np.zeros((n, max_times))
    theta = np.zeros(n)  #参数初始化
    delta_theta = np.ones(n) * h
    llh = 0
    llh_tmp = 0
    b = np.zeros((n, m))
    for i in range(max_times):
        theta_temp = theta
        for j in range(n):
            # 求偏导
            theta[j] += delta_theta[j]
            llh_tmp = likelihood(X, y, theta)
            delta_theta[j] = -h * (llh_tmp - llh) / delta_theta[j]
            #b[j, i] = theta[j]
        theta = theta_temp + delta_theta
        llh = likelihood(X, y, theta)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(llh_tmp), llh, 'red')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(' - Error vs. Iteration')
    plt.show()


    return theta


#预测函数
def predict(X, theta):

    m, n = np.shape(X)
    y = np.zeros(m)
    for i in range(m):
        if sigmoid(X[i], theta) > 0.5: y[i] = 1;
    return y
    return theta

def accuracy(X_test, y_test):
    y_pred = predict(X_test, theta)
    return accuracy_score(y_test, y_pred)

# X_train, X_test, y_train, y_test
np.ones(n)
m, n = np.shape(X)
X_ex = np.c_[X, np.ones(m)]  # 扩展矩阵为 [x, 1]
#print (X_ex)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_ex, y, test_size=0.5, random_state=0)
# 通过梯度下降法得到最优参数
theta = gradDscent_1(X_train, y_train)
# 做出预测 映射theta
y_pred = predict(X_test, theta)
m_test = np.shape(X_test)[0]
# 混淆矩阵的计算和预测精度
cfmat = np.zeros((2, 2))
for i in range(m_test):
    if y_pred[i] == y_test[i] == 0:
        cfmat[0, 0] += 1
    elif y_pred[i] == y_test[i] == 1:
        cfmat[1, 1] += 1
    elif y_pred[i] == 0:
        cfmat[1, 0] += 1
    elif y_pred[i] == 1:
        cfmat[0, 1] += 1
print(cfmat)
acc = accuracy(X_test, y_test)
print(acc)#预测准确率

