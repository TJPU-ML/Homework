#导入四个包
import numpy  as  np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#将三个类转化为0，1，2
def iris_type(s):
    class_label = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    return class_label[s]
#定义一个sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#返回sigmoid函数预测结果值
def model(X, theta):
    # theta = theta.reshape(-1, 1)
    return sigmoid(np.dot(X, theta))

# #根据参数计算损失
def cost(X, y, theta):
    forward = np.multiply(-y, np.log(model(X, theta)))
    backward = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(forward - backward) / (len(X))

#计算每一个参数的梯度方向
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for i in range(len(theta)):
        # print(X[:i])
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad
def descent(X_train, y_train, theta, alpha):
    # 梯度下降求解
    for count in range(1000):
        grad = gradient(X_train, y_train, theta)
        theta = theta - alpha * grad  # 参数更新
    return theta


#得到预测值y
def predict(X, theta):
    y = model(X, theta)
    for i in range(len(y)):
        if y[i] >= 0.5:
            y[i] = 1
        else:
            y[i] = 0
    return y
#图形可视化
X = [x[0] for x in new_iris]
Y = [x[1] for x in new_iris]
plt.scatter(X[:50], Y[:50], color='red', marker='o', label='setosa') #前50个样本
plt.scatter(X[50:100], Y[50:100], color='blue', marker='x', label='versicolor') #中间50个
plt.scatter(X[100:], Y[100:],color='green', marker='+', label='Virginica') #后50个样本
plt.legend(loc=2) #左上角
plt.show()
iris_path = 'iris.csv'#读取数据集
iris = pd.read_csv(iris_path)
new_iris = pd.read_csv(iris_path, converters={5: iris_type})
new_iris = np.array(new_iris)#转化为数组
x = new_iris[0:100, 1:-1]
y = new_iris[0:100, -1]#取前一百行数据，可以正好把数据分为二类花
#分测试集和训练集，比率为3：7
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
temp = np.array(y_train)
temp = temp.reshape(-1, 1)
# print(np.column_stack((np.ones((70)), X_train)))
theta = np.ones((5))
alpha = 0.1
#增加一列1的数据集
X_train = np.column_stack((np.ones((70)), X_train))
# print(X_train)
theta = descent(X_train, y_train, theta, alpha)
theta = theta.reshape(-1, 1)
X_test = np.column_stack((np.ones((30)), X_test))
scaled_X = X_test
y = y_test#得到测试的y_test
predictions = predict(scaled_X, theta)
y = y.reshape(-1, 1)#得到预测的y
# print(y,y_test)
sum= [y==predictions]
print(sum)
j=sum.count('[True]')
print(30-j)
accuray=j/(len(y))
print (1-accuray,theta)
#画出损失函数的图形
costs = [cost(X_train, y_train, theta)]
# for count in range(1000):
#     costs.append(cost(X_train, y_train, theta
ax = plt.subplots(figsize=(12,4))
fig= plt.subplots(figsize=(12,4))
ax.plot(np.arange(len(costs)), costs, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title( 'Error vs. Iteration')