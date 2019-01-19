
import numpy as np #引入数值计算库
import matplotlib.pyplot as plt#引入2D绘图库
import pandas as pd
import time


#函数定义

def sigmoid(z):# sigmoid函数
    return 1 / (1 + np.exp(-z))
def model(X, theta):# 定义回归模型
    return sigmoid(np.dot(X, theta.T))
def Loss_function(X, y, theta):# 定义损失函数
    return np.sum(np.multiply(-y, np.log(model(X, theta))) - np.multiply(1 - y, np.log(1 - model(X, theta)))) / (len(X))
def Gradient(X, y, theta):# 计算梯度
    gradient = np.zeros(theta.shape)
    for k in range(len(theta.ravel())): 
        #梯度公式，见课本
        gradient[0, k] = np.sum( np.multiply((model(X, theta) - y).ravel(), X[:, k])) / len(X)
    return gradient
def stop_falling(type, value, threshold):#梯度下降停止
    if type == 0:
        return value > threshold

# 定义梯度下降求解函数
def descent(data, theta, Size, Type, thresh, alpha):
    init_time = time.time()
    i = 0 
    k = 0  
    X = data[:100, 0:3]
    y = data[:100, 3:]
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [Loss_function(X, y, theta)]  # 计算损失函数
    
    while True:#循环执行
        grad = Gradient(X[k:k + Size], y[k:k + Size], theta)
        k += Size  
        if k >= 100:
            k = 0
            X = data[:100, 0:3]
            y = data[:100, 3:]  
        
        theta = theta - alpha * grad  #梯度下降更新
       
        cost_new = Loss_function(X, y, theta)
        
        costs.append(cost_new)  # 重新计算损失函数的值
        i += 1        
        value = i
        if stop_falling(Type, value, thresh):
            break
    return theta, i - 1, costs, grad, time.time() - init_time
def classification(X, theta):# 设定阈值进行分类
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]
def Iteration(data, theta, Size, stopType, thresh, alpha):# 绘制损失函数图像
    theta, iter, costs, grad, dur = descent(data, theta, Size, stopType, thresh, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} -".format(alpha)
 
    if Size == 100:
        strDescType = "Gradient"
    elif Size == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch({})".format(Size)
    name += strDescType + " descent - stop: "
    if stopType == 0:
        strStop = "{} iterations".format(thresh)
   
    name += strStop
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(costs)), costs, 'b')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta



#处理鸢尾花数据集
iris = pd.read_csv('irisdata'+'//'+'iris2.csv', header=None, names=['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class'], dtype={'sepal_len': float, 'sepal_width': float, 'petal_len': float, 'petal_width': float,'class': float})
iris.insert(2, 'Ones', 1)  # 插入1
Irisdata = iris.as_matrix()  # 得到鸢尾花数据集矩阵
data_setting = Irisdata[:100, :]
np.random.shuffle(data_setting)#将初始数据集打乱
test_num=0
test_num=int(input("选择测试集样本个数："))
test_data = data_setting[0:test_num,2:Irisdata.shape[1] ]
train_data = data_setting[test_num:100,2:Irisdata.shape[1] ]
X = train_data[:100, 1:Irisdata.shape[1]  - 1]
y = train_data[:100, Irisdata.shape[1]  - 1:Irisdata.shape[1] ]

theta = np.zeros([1, 3])#初始theta为0
theta = Iteration(train_data, theta, 100, 0, 50000, 0.05)#学习率为0.05，进行50000次迭代
X_test= test_data[:, :3]#取出前三列数据
y_test= test_data[:, 3]#取出实际待测实际结果
print("theta的值为{}".format(theta))
print("测试集实际值为：")
print(y_test)
result = classification(X_test, theta)
print("测试集预测值为：")
print(result)

num=0#初始化变量
for (a, b) in zip(result, y_test):#统计分类正确个数
    if a==b:
        num=num+1
        
print("测试样本个数为{0}个".format(test_data.shape[0]))
print("测试值与实际值相同的有{0}个".format(num))
correct_rate = (num / test_data.shape[0]) *100#求出正确率
print('所以{0}个训练样本的训练精度为 {1:.2f}%'.format(test_data.shape[0],correct_rate))#输出正确率
