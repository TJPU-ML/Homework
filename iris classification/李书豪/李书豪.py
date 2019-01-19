#导入相应的包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accu

train_size = 0.5    #训练集比例

#sigmoid函数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#预测结果函数
def y_hat(x,theta):
    return sigmoid(np.dot(x,theta.T))

#损失函数
def cost_function(x, y, theta):
    a = y * np.log(y_hat(x, theta))
    b = (1 - y) * np.log(1 - y_hat(x, theta))
    return -np.sum(a + b) / len(x)

def train(x_train,y_train,alpha=0.01):          #训练函数，返回参数theta

    def gradient(x, y, theta):          #计算梯度
        return np.dot(x.T, sigmoid(np.dot(x,theta))-y)/len(x)

    def gd(x, y, initial_theta, alpha):    #使用梯度下降法，得到参数theta
        theta = initial_theta  # theta初始值
        count = 1  # 迭代次数
        max_iter = 1e4
        costs = [cost_function(x, y, theta)]
        while count < max_iter:
            grad = gradient(x, y, theta)      #对参数进行更新
            next_theta = theta - alpha * grad
            if (abs(cost_function(x, y, next_theta) - cost_function(x, y, theta)) < 1e-5):
                break
            theta = next_theta
            costs.append(cost_function(x, y, theta))
            count += 1
        #损失函数变化图
        plt.subplots(figsize=(8, 5))
        plt.plot(np.arange(len(costs)), costs, color='red')
        plt.xlabel('train_num')
        plt.ylabel('Costs')
        plt.title('Costs about train_num', fontsize=20)
        plt.show()
        return next_theta

    x = np.hstack([np.ones((len(x_train), 1)), x_train])     #对x_train在最左侧增加一列1，便于矩阵计算
    initial_theta = np.zeros(x.shape[1])
    Theta = gd(x, y_train, initial_theta, alpha)
    return Theta

def predict(x_predict):     #结果预测，result>=0.5时归为'1'，其余归为'0'
    x = np.hstack([np.ones((len(x_predict), 1)), x_predict])    #对x_predict在最左侧增加一列1，便于矩阵计算
    theta = train(x_train,y_train)     #调用train函数训练得到参数theta，用于结果预测
    result = y_hat(x, theta)
    return np.array(result >= 0.5, dtype='int')

def accuracy(x_test,y_test):         #计算准确率
    y_predict = predict(x_test)
    return accu(y_test,y_predict)   #调用acuuracy_score函数计算准确率

my_data = pd.read_csv("iris.csv",index_col=0,usecols=[0,1,2,5])
my_data.loc[my_data.Species=='setosa','Species'] = 0          #将鸢尾类别数据化为setosa:0,versicilor:1
my_data.loc[my_data.Species=='versicolor','Species'] = 1
data = np.array(my_data)
x, y = data[:, :-1], data[:, -1]        #x取前两列，y取最后一列

#散点图
setosa = my_data[my_data['Species'] == 1]
versicolor = my_data[my_data['Species'] == 0]
plt.subplots(figsize=(8,5))
plt.scatter(setosa['Sepal.Length'], setosa['Sepal.Width'], s=30, c='red', marker='D', label='setosa')
plt.scatter(versicolor['Sepal.Length'], versicolor['Sepal.Width'], s=30, c='green', marker='o', label='versicolor')
plt.xlabel('Sepal.Length')    #设置x轴为Sepal.Length，y轴为Sepal.Width
plt.ylabel('Sepal.Width')
plt.legend()
plt.title(r'Scatter plot', fontsize=20)
plt.show()

#训练集属性  测试集属性  训练集标签  测试集标签
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_size, random_state=120)

acc = accuracy(x_test, y_test)
print('正确率:', acc)






