import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#读取数据集
Iris = pd.read_csv('iris.csv')
#sigmod函数
def sigmoid(x1,x2, theta_1, theta_2,theta_3):
    z = (theta_1*x1+ theta_2*x2+theta_3).astype("float_")
    return 1.0 / (1.0 + np.exp(-z))
#下降的梯度
def gradient(x1,x2, y, theta_1, theta_2,theta_3):
    sigmoid_probs = sigmoid(x1,x2,theta_1, theta_2,theta_3)
    return 1/len(y)*np.sum((y - sigmoid_probs)*x1),\
1/len(y)*np.sum((y - sigmoid_probs)*x2),\
1/len(y)*np.sum((y - sigmoid_probs))


# 梯度下降法
def GradDe(x1, x2, y, Max_Loop, alpha):
    theta_1 = 0.1
    theta_2 = 0.5
    theta_3 = 0.56
    error = np.ones((Max_Loop,))
    J_history = np.zeros(Max_Loop)
    for i in range(Max_Loop):
        delta1, delta2, delta3 = gradient(x1, x2, y, theta_1, theta_2, theta_3)

        theta_1 = theta_1 + alpha * delta1
        theta_2 = theta_2 + alpha * delta2
        theta_3 = theta_3 + alpha * delta3

        y_pre = predict(test_x1, test_x2, theta_1, theta_2, theta_3)
        J_history[i] = Cost(train_x1, train_x2, y_train, theta_1, theta_2, theta_3)
        y_newpre = np.around(y_pre)  # 四舍五入后的值，大于0.5为1，小于0.5为0

        if i % 100 == 0:
            print('delta%d =' % (i), [delta1, delta2, delta3])
            print('theta%d =' % (i), [theta_1, theta_2, theta_3], '\n')
    y_pred = np.around(y_pre)
    acc_score = accuracy_score(y_test, y_pred)
    print('准确率：', acc_score)
    return [theta_1, theta_2, theta_3, J_history]
#预测函数
def predict(x1,x2, theta_1, theta_2, theta_3):
    y_pre=sigmoid(x1,x2, theta_1, theta_2,theta_3)
    return y_pre
def Cost(x1,x2,y_train,theta_1, theta_2,theta_3):
    n=y_train.size
    J=(1.0/(2*n))*np.sum(np.square(sigmoid(x1,x2, theta_1, theta_2,theta_3)-y_train))
    return J


if __name__ == "__main__":
    # 将数据划分成0,1两种类型
    class_mapping = {label: idx for idx, label in enumerate(np.unique(Iris['Species']))}
    Iris['Species'] = Iris['Species'].map(class_mapping)

    Species_mapping = {
        0: 1,
        1: 0,
        2: 0
    }
    Iris['Species'] = Iris['Species'].map(Species_mapping)
    # 划分测试集与训练集
    x, y = Iris[['Petal.Length', 'Petal.Width']], Iris['Species']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    train_x1 = x_train['Petal.Length']
    train_x2 = x_train['Petal.Width']
    test_x1 = x_test['Petal.Length']
    test_x2 = x_test['Petal.Width']
    # 梯度下降

    theta_1, theta_2, theta_3, J_history = GradDe(train_x1, train_x2, y, 1000, 0.1)

    y_pre = predict(test_x1, test_x2, theta_1, theta_2, theta_3)
    plt.xlabel('Number of training')
    plt.ylabel('Error rate')
    plt.plot(J_history)
    # 获取属性数据
    features = x.values
    # 获取分类信息
    classInfo = y

    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(classInfo)):
        if classInfo[i] == 0:
            type1_x.append(features[i][0])
            type1_y.append(features[i][1])
        if classInfo[i] == 1:
            type2_x.append(features[i][0])
            type2_y.append(features[i][1])
        # 数据绘图   
    type1 = ax.scatter(type1_x, type1_y, s=30, c='g')
    type2 = ax.scatter(type2_x, type2_y, s=30, c='r')

    _x = np.arange(1.5, 3.5, 0.1)
    _y = (-theta_3 - theta_1 * _x) / theta_2
    ax.plot(_x, _y)
    # 设置横坐标和纵坐标
    plt.xlabel('Petal.Length')
    plt.ylabel('Petal.Width')
    plt.show()