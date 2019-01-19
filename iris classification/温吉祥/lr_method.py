# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 05:08:00 2018

@author: WinJX
"""

import numpy as np

'''
最大似然项函数，课本3.27(page 59)
'''
def theta_sub(x, y, beta):
    return -y * np.dot(beta, x.T) + np.math.log(1 + np.math.exp(np.dot(beta, x.T)))   

'''
最大似然项求和函数，课本3.27(page 59)   
'''
def theta(X, y, beta):
    sum = 0
    m,n = np.shape(X)      
    for i in range(m):
        sum += theta_sub(X[i], y[i], beta)                                                 
    return sum       

'''
梯度下降函数，不断迭代返回最优参数值beta
'''
def grad_desc(X, y): 
    h = 0.05                       #学习率
    max_iters= 300                 # 最大迭代次数  
    m, n = np.shape(X)             #(m,n) == (100,3)            
    beta = np.zeros(n)             # 初始化beta值 = [0,0,0]  
    delta_beta = np.ones(n)*h      #初始化梯度delta_beta为学习率h大小
    tha = 0                        
    tha_temp = 0   
    for i in range(max_iters):
        beta_temp = beta.copy()       
        for j in range(n): 
            beta[j] += delta_beta[j]       #梯度delta_beta         
            tha_temp = theta(X, y, beta)   
            delta_beta[j] = -h * (tha_temp - tha) / delta_beta[j]     #课本408页（B.17）                
            beta[j] = beta_temp[j]            
        beta += delta_beta            
        tha = theta(X, y, beta)        #当前参数下的损失函数值 tha
        
    print('该模型的损失函数值为：%.2f'%tha)              
    return beta

'''
对数几率函数，课本3.18
'''
def sigmoid(x, beta):
    return 1.0 / (1 + np.math.exp(- np.dot(beta, x.T)))  
    
'''
预测函数，用测试集和获得的beta值带入sigmoid函数得到介于0-1的预测值y,
           以0.5作为阈值，大于0.5输出1，反之输出0
'''
def predict(X, beta):
    m, n = np.shape(X)
    y = np.zeros(m)   
    for i in range(m):
        if sigmoid(X[i], beta) > 0.5:
            y[i] = 1
        else:
            y[i] = 0
    return y                             
    return beta
