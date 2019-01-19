import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
iris=sns.load_dataset("iris")
#sigmoid函数
def sigmoid(X):
    return 1/(1+np.exp(-X))
# 预测
def predict(X_train,theta):
    return sigmoid( X_train * theta )
# 获取错误率
def predict_error(X_train,X_test,Y_train,Y_test,theta):
    train_pred = np.round(predict(X_train,theta))
    test_pred = np.round(predict(X_test,theta))
    error_train = 1- metrics.accuracy_score(Y_train, train_pred)
    error_test = 1- metrics.accuracy_score(Y_test, test_pred)
    return error_train,error_test
#梯度下降法
def gra_ascent_train( X ,Y,alpha=0.01,max_iter=100, test_size = 0.3,random_state= 0):
    
    X_train,X_test,Y_train,Y_test = train_test_split(
        X ,Y, test_size=test_size, random_state= random_state)
#     初始化
    theta = np.ones((X_train.shape[1],Y_train.shape[1]))
    error_train = np.ones((max_iter,))
    error_test = np.ones((max_iter,))
#     训练
    for i in range(max_iter):
        error_train[i],error_test[i] = predict_error(X_train,X_test,Y_train,Y_test,theta)
        
        delta = predict(X_train,theta) - Y_train
        theta = theta - alpha * X_train.transpose() * delta
    return theta,error_train,error_test

#对类型进行重编码（one-hot）
dummies_iris = pd.get_dummies(iris['species'], prefix= 'species')
iris_df = pd.concat([iris, dummies_iris], axis=1)
iris_df.drop(['species'], axis=1, inplace=True)
iris_df.describe()
Y = iris_df.iloc[:100,4:5]
#print(Y)
X =iris_df.iloc[:100,2:4]
X['b'] = 1
#print(X)
# 求解
for i,test_sizei in enumerate([0.1,0.3,0.5]):
    theta,error_train,error_test = gra_ascent_train(np.mat(X) ,np.mat(Y),alpha=0.001,max_iter=2000,test_size = test_sizei,random_state= 111)
    print("test_size:{0:.1f}|  accuracy_score_train: {1:.3f}  accuracy_score_test:{2:.3f}".format(test_sizei,1-error_train[-1:][0],1-error_test[-1:][0]))
   
