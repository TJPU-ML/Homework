from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    '''
    计算激活函数后的值
    :param x: 输入向量
    :return: 经激活后的输出向量
    '''
    return 1 / (1 + np.exp(-x))


def logistic_regression_train(features, labels, iters, alpha):
    '''
    Logistic模型训练
    :param features: 特征矩阵
    :param labels: 标签向量
    :param iters: 最大迭代次数
    :param alpha: 学习率
    :return: 权重、损失函数值向量
    '''
    n = np.shape(features)[1]
    w = np.mat(np.random.rand(n, 1))

    cost = []
    i = 0
    while i < iters:
        i += 1
        t = features * w
        h = sigmoid(t)
        err = labels - h
        w = w + alpha * features.T * err / features.shape[0]
        c = get_cost(features*w, labels)
        cost.append(c.tolist()[0])
        print('迭代次数 = %d, 损失 = %f.' % (i, c))

    return (w, cost)

def get_cost(h, labels):
    '''
    计算损失函数值
    :param h: 输入向量
    :param labels: 真实标签
    :return: 损失函数值
    '''
    sum = 0

    for i in range(h.shape[0]):
        sum += labels[i] * np.math.log(sigmoid(h[i])) + (1-labels[i]) * np.math.log(1-sigmoid(h[i]))
    return -sum / h.shape[0]

def get_right_rate(predict, real):
    '''
    获取训练过程中正确率
    :param predict: 预测值向量
    :param real: 真实值向量
    :return: 正确率
    '''
    sum = 0
    for i in range(predict.shape[0]):
        if predict[i] == real[i]:
            sum += 1
    return sum/predict.shape[0]


def predict(features, w):
    '''
    预测
    :param feature: 特征矩阵
    :param w: 权重
    :return: 预测值
    '''
    h = sigmoid(features * w)
    p = np.where(h>0.5, 1. , 0.)

    return p

def main():
    train_rate = 0.7  # 训练集占比
    iters_max = 2000  # 最大迭代次数
    alpha = 0.01  # 学习率

    data = datasets.load_iris()  # 此处使用sklearn包导入鸢尾花数据集

    features_0 = data['data'][50:100]  # Versicolour特征
    features_1 = data['data'][100:150]  # Virginica特征

    ax = plt.subplot(111, projection='3d')
    ax.scatter(features_0[:, 1], features_0[:, 2], features_0[:, 3], c = 'b')
    ax.scatter(features_1[:, 1], features_1[:, 2], features_1[:, 3], c = 'r')

    plt.show()

    train_n = int(train_rate*50)  # 训练集数目
    test_n = 50 - train_n  # 测试集数目

    labels_0 = np.zeros((50, 1))
    labels_1 = np.ones((50, 1))
    features = np.vstack((features_0[0:train_n], features_1[0:train_n]))
    features = np.hstack((np.ones((features.shape[0], 1)), features))  # 训练集特征
    labels = np.vstack((labels_0[0:train_n], labels_1[0:train_n]))  # 训练集标签
    features_test = np.vstack((features_0[-test_n:], features_1[-test_n:]))
    features_test = np.hstack((np.ones((features_test.shape[0], 1)), features_test))  # 测试集特征
    labels_test = np.vstack((labels_0[-test_n:], labels_1[-test_n:]))  # 测试集标签

    (w, cost) = logistic_regression_train(features, labels, iters_max, alpha)  # 训练

    print('最佳的权重为：')
    print(w)
    r = get_right_rate(predict(features_test, w), labels_test)
    print('正确率：')
    print(r)

    plt.plot(np.linspace(1, len(cost), len(cost)), cost)
    plt.title('Loss function')
    plt.show()


if __name__ == '__main__':
    main()
