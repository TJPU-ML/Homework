# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 05:06:40 2018

@author: WinJX
"""

import lr_method
from sklearn.datasets import load_iris
import numpy as np 
from sklearn import model_selection

'''
导入数据，其中X(共有三类及四个属性，选取前两类及前两个属性）为数据集，y为所对应标记（0 or 1)
'''
data_need = load_iris()       #获取原始数据                                                          
X = data_need.data[:100,:2]   #取数据的前两类和前两个属性
y = data_need.target[:100]    #取前两类数据标记值(0 or 1)

'''
分类训练集和测试集
'''
m,n = np.shape(X)                 #(m,n) == (100,2)
X_hat = np.c_[X, np.ones(m)]    # 扩展矩阵X为X^ = [x,1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_hat, y, test_size=0.5, random_state=1)
                                  #分类训练集合测试集，test_size为分类比例 
beta = lr_method.grad_desc(X_train, y_train)
                                  #调用梯度下降函数获得β值，beta = [w,b]
y_pret = lr_method.predict(X_test, beta)
                                  #用获得的beta值及测试集求得预测值y_pret
                                  
'''
结果统计
'''
m_test = np.shape(X_test)[0]       # m_test值等于测试集大小
count = 0            #用于统计预测正确的个数             
ret_target = []      #用列表输出每个数据的正确与否，‘1’表示预测正确，‘0’表示预测错误
for i in range(m_test):   
    if y_pret[i] == y_test[i]: 
        count += 1
        ret_target += '1'
    else:
        ret_target += '0'
accuracy = (count/m_test)       #精度                     
print('预测情况 = ',ret_target)
print("该模型在{0}个测试集上的精度为{1:.2%}".format(m_test,accuracy))
