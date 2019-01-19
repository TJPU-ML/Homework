#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D


def sig(x):
    '''
    :x: 输入向量
    :return: 输出向量
    '''
    return 1 / (1 + np.exp(-x))

def train(feature_matrix, label_vector, max_degree, learning_rate):
    '''
    对数几率回归训练模型
    :feature_matrix: 特征矩阵
    :label_vector: 标签向量
    :max_degree: 最大迭代次数
    :learning_rate: 学习率
    :return: 权重、损失函数值向量
    '''
    i = 0
    cost = []
    n = np.shape(feature_matrix)[1]
    w = np.mat(np.random.rand(n, 1))
    while i < max_degree:
        i += 1
        t = feature_matrix * w
        h = sig(t)
        err = label_vector - h
        w = w + learning_rate * feature_matrix.T * err / feature_matrix.shape[0]
        c = get_cost(feature_matrix*w, label_vector)
        cost.append(c.tolist()[0])
        print('iterations = %d, loss = %f.' % (i, c))
    return (w, cost)

def get_cost(vector, real_labels):
    '''
    计算损失函数值
    :vector: 输入向量
    :real_labels: 真实标签
    :return: 损失函数值
    '''
    count = 0
    for i in range(vector.shape[0]):
        count += real_labels[i] * np.math.log(sig(vector[i])) + (1-real_labels[i]) * np.math.log(1-sig(vector[i]))
    return -count / vector.shape[0]

def right_rate(predict_vector, real_vector):
    '''
    获取训练过程中正确率
    :predict_vector: 预测值向量
    :real_vector: 真实值向量
    :return: 正确率
    '''
    count = 0
    for i in range(predict_vector.shape[0]):
        if predict_vector[i] == real_vector[i]:
            count += 1
    return count/predict_vector.shape[0]


def predict(feature_vector, weight):
    '''
    :feature_vector: 特征矩阵
    :weight: 权重
    :return: 预测值
    '''
    h = sig(feature_vector * weight)
    p = np.array(list(map(lambda x: int(x>0.5), h)))
    return p

def main():
    x = np.arange(-10, 10, 0.1)
    h = sig(x)  # Sigmoid函数
    plt.plot(x, h)
    plt.axvline(0.0, color='k')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])  # y axis label
    plt.title(r'Sigmoid函数曲线', fontsize=15)
    plt.text(5, 0.8, r'$y = \frac{1}{1+e^{-z}}$', fontsize=18)
    plt.show()
    train_rate = 0.1
    max_degree = 2000
    learning_rate = 0.001
    # sklearn包导入鸢尾花数据集
    data = datasets.load_iris()
    features_0 = data['data'][50:100]
    features_1 = data['data'][100:150]
    ph = plt.subplot(111, projection='3d')
    ph.scatter(features_0[:, 1], features_0[:, 2], features_0[:, 3], c = 'b')
    ph.scatter(features_1[:, 1], features_1[:, 2], features_1[:, 3], c = 'r')
    plt.show()
    label_vector_0 = np.mat(data['target'][0:50]).T
    label_vector_1 = np.mat(data['target'][50:50 * 2]).T
    train_n = int(train_rate*50)
    test_n = 50 - train_n
    label_vector = np.vstack((label_vector_0[0:train_n], label_vector_1[0:train_n]))
    feature_matrix = np.vstack((features_0[0:train_n], features_1[0:train_n]))
    feature_matrix = np.hstack((np.ones((feature_matrix.shape[0], 1)), feature_matrix))
    features_test = np.vstack((features_0[-test_n:], features_1[-test_n:]))
    features_test = np.hstack((np.ones((features_test.shape[0], 1)), features_test))
    labels_test = np.vstack((label_vector_0[-test_n:], label_vector_1[-test_n:]))
    # 开始训练
    (w, cost) = train(feature_matrix, label_vector, max_degree, learning_rate)
    print('optimum_weight:')
    print(w)
    r = right_rate(predict(features_test, w), labels_test)
    print('right_rate：%f' %(r))
    plt.plot(np.linspace(1, len(cost), len(cost)), cost)
    plt.title('Loss function')
    plt.show()

if __name__ == '__main__':
    main()