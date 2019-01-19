import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#读取文件，并转换成矩阵形式
def datalist(filename):
    file=open(filename)
    x=[]
    y=[]
    for line in file.readlines():
        line=line.strip().split()
        x.append([1,float(line[3]),float(line[4])])
        y.append(float(line[5]))
    xmat=np.mat(x)
    ymat=np.mat(y).T
    file.close()
    return xmat,ymat

xmat,ymat= datalist('iris.txt')
#print('xmat:',xmat,xmat.shape)
#print('ymat:',ymat,ymat.shape)

#计算函数
def calc(xmat,ymat,alpha=0.001,maxIter=20000):
    W=np.mat(np.random.randn(3,1))
    cost_fuction,maxIter_times=[],[]
    for i in range(maxIter):
        H=1/(1+np.exp(-xmat*W)) #sigmoid
        dw=xmat.T*(H-ymat)
        W-= alpha*dw  #梯度下降法
        if i%200 == 0:
            cost = -ymat.T * np.log(H) - (1 - ymat).T * np.log(1 - H)  # 损失函数
            cost_fuction.append(cost.tolist()[0][0])
            maxIter_times.append(i)#迭代次数
    return W,cost_fuction,maxIter_times

W,cost_fuction,maxIter_times=calc(xmat,ymat)
#print('W:',W)
#print('cost:',cost)

#绘制散点图，分界线
w0=W[0,0]
w1=W[1,0]
w2=W[2,0]
plotx1=np.arange(1,7,0.01)
plotx2=-w0/w2-w1/w2*plotx1
plt.plot(plotx1,plotx2,c='r')
plt.scatter(xmat[:,1][ymat==0].A,xmat[:,2][ymat==0].A,label='setosa')
plt.scatter(xmat[:,1][ymat==1].A,xmat[:,2][ymat==1].A,label='versicolor')
plt.grid()
plt.legend()
plt.show()

#绘制损失函数图像
plt.plot(maxIter_times,cost_fuction)
plt.show()


