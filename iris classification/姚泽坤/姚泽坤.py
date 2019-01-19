from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Iris-setosa=0;Iris-versicolor=1
TRAIN_SIZE=0.9


#加载数据集
data=loadtxt(open("iris.txt"),delimiter=',',skiprows=0)
X,y=data[:,:-1],data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-TRAIN_SIZE, random_state=42)
X_train, X_test=mat(X_train),mat(X_test)
y_train, y_test=mat(y_train).T,mat(y_test).T

w1=mat(random.rand(4,1))
b1=mat(zeros((1,1)))
B=vstack((w1,b1))
m,n=X_train.shape
x1=mat(ones((m,1)))
m_,n_=X_test.shape
x2=mat(ones((m_,1)))
#print(X_train.shape, x1.shape)
X_TRAIN=hstack((X_train, x1))
X_TEST=hstack((X_test, x2))

#定义sigmoid函数
def sigmoid(X):
    return(1/(1+exp(-X)))

#前向传播
def forward(X,B):
    y=sigmoid(X*B)
    return y

'''
def FORWARD(X,B):
    y=X*B
    return y
'''

#定义损失函数
def loss(X,B,y):
    return sum(multiply(-y,log(forward(X,B)))-multiply((1-y),log(1-forward(X,B))))/(len(X))

#定义梯度下降法
def SGD(X,B,y,learning_rate,maxIter):
    pic_x=[]
    pic_y1=[]
    pic_y2=[]
    for i in range(maxIter):
        #计算梯度：
        dB=X.T*(forward(X,B)-y)
        #梯度下降：
        B-=learning_rate*dB
        #绘制图线
        if (i%10==0):
            pic_x.append(i)
            pic_y1.append(loss(X,B,y))
            pic_y2.append(test(X_TEST,B,y_test))

    plt.plot(pic_x,pic_y1)
    plt.show()
    plt.plot(pic_x,pic_y2)
    plt.show()
    return B


#定义测试函数
def test(X,B,y):
    y_=forward(X,B)
    #print(y_)
    #将概率值改为0,1
    Change=where(y_>=0.5)
    y_[Change]=1
    Change=where(y_<0.5)
    y_[Change]=0
    #print(y_)
    #挑选相同元素
    Y=y_-y
    #print(Y)
    #print(len(Y))
    #将与标签相同的筛选出来另成一个矩阵
    number=mat(Y[where(Y==0)]).T
    #print(len(number))
    right_rate=len(number)/len(Y)
    #print("正确率为%f"%right_rate)
    return right_rate

#test(X_TEST,B,y_test)
B=SGD(X_TRAIN,B,y_train,0.001,500)
w1=B[:-1,:]
b1=B[-1,:]
print(w1)
print(b1)
print(test(X_TEST,B,y_test))
#print(loss(X,B,y_train))
