# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 损失函数可视化坐标
cost_arr = []
iter_arr = []


# 读入数据
def loaddata(filename):
    # 打开文件 放入file
    file = open(filename)

    x = []
    y = []
    # 将file中的每一行read 放入line
    for line in file.readlines():
        # 预处理line 去掉两边的空格 并分成三部分
        line = line.strip().split()
        # 换为整型
        x.append([1, float(line[0]), float(line[1])])
        y.append(float(line[-1]))
        # 数据集划分
        #X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5, random_state=0)
        #X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    xmat = np.mat(X_train)
    ymat = np.mat(Y_train).T
    xmat_T = np.mat(X_test)
    ymat_T = np.mat(Y_test).T

    # 关闭文件
    file.close()
    return xmat, ymat,xmat_T,ymat_T


# 求解W和cost
def w_calc(xmat, ymat, alpha=0.001, maxIter=10001):
    # 初始化W 三行一列 randn服从正态分布
    W = np.mat(np.random.randn(3, 1))

    # 更新W和cost 并将cost值存入数组 便于绘图
    for i in range(maxIter):
        H = 1 / (1 + np.exp(-xmat * W))
        dw = xmat.T * (H - ymat)  # dw:(3,1)
        W -= alpha * dw
        cost = (-ymat.T * np.log(H) - (1 - ymat.T) * np.log(1 - H))
        cost_arr.append(np.max(cost))
        iter_arr.append(i)
    return W

# 测试正确率
def test(xmat_T,ymat_T):
    H_T=1/(1+np.exp(-xmat_T*W))
    #由函数图像得 函数值>=0.5则标记为label=1 否则为0
    H_T [np.where(H_T >= 0.5)] = 1
    H_T[np.where(H_T < 0.5)]= 0
    #作差 差为0 则预测正确
    difference=H_T-ymat_T
    num = difference[np.where(difference == 0)].T
    accuracy_rate = len(num) / len(difference)
    return accuracy_rate

# 运行
xmat, ymat,xmat_T,ymat_T = loaddata('iris.txt')
W = w_calc(xmat, ymat, 0.001, 10000)
print('W:', W)
accuracy_rate = test(xmat_T,ymat_T)
print('accuracy_rate =',accuracy_rate)
# 可视化数据分布
w0 = W[0, 0]
w1 = W[1, 0]
w2 = W[2, 0]
plotx1 = np.arange(1, 7, 0.01)
plotx2 = -w0 / w2 - w1 / w2 * plotx1
plt.plot(plotx1, plotx2, c='r')
# 输出测试集所有坐标点 显示label 布尔运算 取ymat=0
plt.scatter(xmat_T[:, 1][ymat_T == 0].A, xmat_T[:, 2][ymat_T == 0].A, label='label=setosa', marker='s')
#  取ymat=1
plt.scatter(xmat_T[:, 1][ymat_T == 1].A, xmat_T[:, 2][ymat_T == 1].A, label='label=versicolor')
# 显示网格
plt.grid()
# 显示图例
plt.legend()
plt.show()

# 可视化损失函数
plotx3 = iter_arr
plotx4 = cost_arr
plt.plot(plotx3, plotx4)
plt.show()
