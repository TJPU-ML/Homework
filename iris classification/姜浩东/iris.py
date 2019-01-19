import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

TRAIN_SIZE=0.5


#加载数据
data=np.loadtxt(open("iris.txt"),delimiter=',',skiprows=0)
X,y=data[:,:-1],data[:,-1]
plt.scatter(X[:50,:1],X[50:100,1:2],color = 'red')
plt.scatter(X[50:100,:1],X[50:100,1:2],color = 'blue')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-TRAIN_SIZE, random_state=42)
X_train, X_test=np.mat(X_train),np.mat(X_test)
y_train, y_test=np.mat(y_train).T,np.mat(y_test).T

w1=np.mat(np.random.rand(4,1))
#定义sigmoid函数
def sigmoid(X):
    return(1/(1+np.exp(-X)))

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
    loss=(np.sum(np.multiply(-y,np.log(forward(X,B)))-np.multiply((1-y),np.log(1-forward(X,B)))))/(len(X))
    return loss
#梯度下降法
def SGD(X,B,y,learning_rate,maxIter):
    pic_x=[]
    pic_y1=[]

    for i in range(maxIter):
        #计算梯度：
        dB=X.T*(forward(X,B)-y)
        #梯度下降：
        B-=learning_rate*dB
        #绘制图线
        if (i%10==0):
            pic_x.append(i)
            pic_y1.append(loss(X,B,y))

    plt.plot(pic_x,pic_y1)
    plt.show()
    
    return B





#测试函数
#定义测试函数
def test(X,B,y):
    y_pret=forward(X,B)
  
    #将概率值改为0,1
    Change=np.where(y_pret>=0.5)
    y_pret[Change]=1
    Change=np.where(y_pret<0.5)
    y_pret[Change]=0
    
    #挑选相同元素
    Y=y_pret-y
   
   
    #将与标签相同的筛选出来另成一个矩阵
    number=np.mat(Y[np.where(Y==0)]).T
   
    right_rate=len(number)/len(Y)
  
    return right_rate


B=SGD(X_train,w1,y_train,0.001,150)
rate = test(X_test,w1,y_test)*100
print('模型精度为：%.2f%%'%rate)
