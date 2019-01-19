
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']         # 解决中文乱码问题
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False          # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串
from matplotlib import pyplot as plt



# 加载数据集
def loadDataSet(fileName):
    frTrain = open(fileName)
    trainingSet = []
    trainingLabel = []
    for line in frTrain.readlines():
        curLine = line.strip().split(',')
        lineArr = []
        for i in range(curLine.__len__()-1):
            lineArr.append(float(curLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(curLine[-1]))
    return trainingSet, trainingLabel


# 数据集划分
def divide(fileName):
    dataSet = loadDataSet(fileName)
    X_train, X_test, y_train, y_test = train_test_split(dataSet[0], dataSet[1], test_size=0.1, random_state=0) # 以0为种子随机选择
    return X_train, X_test, y_train, y_test


# 梯度下降法，其中：X(m*n矩阵，m为样本个数，n为特征数)、y（m维列向量）表示输入输出，beta为系数（n维列向量，包括常数项），alpha为学习率，numIterations为迭代次数
def gradDescent(X, y, beta, alpha, numIterations):
    X = np.transpose(X)             # 4 * m , 与西瓜书保持一致
    numIter = []                    # 横坐标
    costArray = []                       # 纵坐标
    gradArray = []
    for count in range(0, numIterations):
        Xbar = np.row_stack((X, np.ones((1, y.size))))
        XbarTrans = np.transpose(Xbar)
        yTemp = np.dot(XbarTrans, beta)              # 线性回归
        yPredict = 1 / (1 + np.exp(-yTemp))   # 对数几率回归
        loss = y - yPredict                 # m * 1
        cost = np.sum(loss ** 2) / (2 * len(y))
        numIter.append(count + 1)
        costArray.append(cost)
        grad = - np.dot(Xbar, loss)
        # gradArray.append(grad)
        beta = beta - alpha * grad
    # 绘制损失函数变化曲线
    plt.figure(1)
    plt.plot(numIter[2:-1], costArray[2:-1])
    plt.ylim(-1e-7, 1e-6)
    # plt.yticks(np.linspace(-0.00005, 0.00001, 0.00005, endpoint=True))
    plt.title('损失函数变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('Loss')
    return beta


# 归类
def classify(X, beta):
    X = np.transpose(X)
    m, n = X.shape
    Xbar = np.row_stack((X, np.ones((1, n))))
    XbarTrans = np.transpose(Xbar)
    yTemp = np.dot(XbarTrans, beta)  # 线性回归
    yPredict = 1 / (1 + np.exp(-yTemp))  # 对数几率回归
    for i in range(0, yPredict.size):
        if yPredict[i] >= 0.5:
            yPredict[i] = 1
        else:
            yPredict[i] = 0
    return yPredict


# 降维，by 主成分分析
def myPCA(data):
    pca = PCA(n_components=2)
    newData = pca.fit_transform(data)
    return newData


## 主函数
# 读取数据集
dataSet1 = divide('Iris-setosa.txt')
dataSet2 = divide('Iris-versicolor.txt')
trainingSet = np.vstack((dataSet1[0], dataSet2[0]))
labelSet = np.hstack((dataSet1[2], dataSet2[2]))
testingSet = np.vstack((dataSet1[1], dataSet2[1]))

# 提取各个成分各自的方差百分比
m, n = trainingSet.shape
pca = PCA(n_components=n)
newData = pca.fit_transform(trainingSet)
print('训练集各个特征各自的方差百分比:', pca.explained_variance_ratio_)

# 主成分分析
newTrainingSet = myPCA(trainingSet)
plt.figure(figsize=(8, 6))
plt.figure(2)
plt.scatter(myPCA(newTrainingSet)[0:int(m/2), 0], myPCA(newTrainingSet)[0:int(m/2), 1], marker='o')
plt.scatter(myPCA(newTrainingSet)[int(m/2)+1:-1, 0], myPCA(newTrainingSet)[int(m/2)+1:-1, 1], marker='x')
plt.title('PCA降维后的散点图')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')

# 求系数beta
X = newTrainingSet
y = labelSet
y = y.reshape(-1, 1)
m, n = np.shape(X)
print(m)
beta = np.ones((n+1, 1))        # beta 初始化为单位向量
numIterations = 5000
alpha = 0.1
beta = gradDescent(X, y, beta, alpha, numIterations)
print(beta)

# 预测
newTestingSet = myPCA(testingSet)
yPredict = classify(newTestingSet, beta)

# 识别率lamda
labelSet = np.hstack((dataSet1[3], dataSet2[3]))
lamda = np.sum(np.transpose(yPredict) == labelSet) / yPredict.size
print('识别正确率为', lamda*100, '%')

# 显示分类散点图
plt.show()
