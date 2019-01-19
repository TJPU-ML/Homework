import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import random as rd
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier
#用0,1替换鸢尾花的种类
def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1}
    return it[s]
  #########################################
#提取鸢尾花数据集
path=u'E:/AI/venv/iris.data'# 数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
#将iris分为训练集与测试集
Data=rd.sample(list(data),50)#选取五十个
Data=np.array(Data)
x1=data[:,0]
x2=data[:,1]
y=data[:,4]
#####################################
y=y.astype(int)
m=len(x1)
#########################################
theta1,theta2,theta3=sp.symbols("theta1 theta2 theta3")
#sigmoid函数
def Sigmoid(x1,x2):
    z=(theta1*x1+theta2*x2+theta3)
    return 1.0/(1.0+np.e**(-z))
S=Sigmoid(x1,x2)
#损失函数
def Cost():
    z2=np.array(list(map(lambda m,n:m*sp.log(n)+(1-m)*sp.log(1-n),y,S)))
    return -1/m*np.sum(z2)
C=Cost()
#梯度下降
def gradient(alpha,theta1_1,theta2_2,theta3_3,theta1,theta2,theta3):
    return sp.diff(C,theta1),\
         sp.diff(C,theta2),\
           sp.diff(C,theta3)
gra1=gradient(theta1)
gra2=gradient(theta2)
gra3=gradient(theta3)
theta1_1=0
theta2_2=0
theta3_3=0
alpha=0.2
number=0
cost_list=[]
while 1:
    number+=1
    temp1 = theta1_1-alpha*gra1.subs([( theta1,theta1_1),(theta2,theta2_2),(theta3,theta3_3)])
    temp2 = theta2_2-alpha*gra2.subs([( theta1,theta1_1),(theta2,theta2_2),(theta3,theta3_3)])
    temp3 = theta3_3-alpha*gra3.subs([(theta1, theta1_1),(theta2,theta2_2),(theta3,theta3_3)])
    theta1_1=temp1
    theta2_2=temp2
    theta3_3=temp3

    cost_now=C.subs([(theta1, theta1_1),(theta2,theta2_2),(theta3,theta3_3)])
    cost_list.append(cost_now)
    if number>10/alpha:
        if abs(cost_list[-2]-cost_list[-1])<0.1:
            break
        print("C:%f"%(cost_now))
        print('theta1=%f,theta2=%f,theta3=%f'%(theta1_1,theta2_2,theta3_3))
#测试
Test1=set([tuple(num) for num in data])
Test2=set([tuple(num) for num in Data])
Test=np.array(list(map(lambda x:list(x),list(Test1-Test2))))
X1=data[:,0]
M=len(X1)
X2=data[:,1]
Y=data[:,4]
Y=Y.astype(int)
###################################
#统计正确率
Error=0
#计算预测值结果 四舍五入后的输出值大于0.5为1，小于0.5为0
End_test=1.0/(1.0+np.e**(-(theta1_1*X1+theta2_2*X2+theta3_3)))
if End_test>=0.5:
    End_test=1
else:
    End_test=0
if (End_test-Y)==0:
  Error=Error
else:
   Error=Error+1
print("测试数据集的正确率为：%f"%((1-Error/M)*100))
#绘制坐标图及函数
plt.xlabel("length")
plt.ylabel("width")
element=np.arange(4,8)
Y2=(theta1_1*element+theta3_3)/(-theta2_2)
plt.plot(element,Y2,color='black')
y = list(y)
Y=list(Y)
x1_test_1=X1[list([i for i,x in enumerate(Y) if x ==1] )]
y1_test_1=X2[list([i for i,x in enumerate(Y) if x ==1] )]
x2_test_0=X1[list([i for i,x in enumerate(Y) if x ==0] )]
y2_test_0=X2[list([i for i,x in enumerate(Y) if x ==0] )]
#打印
plt.plot(x1_test_1,y1_test_1,'go')
plt.plot(x2_test_0,y2_test_0,'bo')
plt.show()
#绘制损失函数
plt.xlabel("frequency")
plt.ylabel("cost")
plt.plot(range(number),cost_list)
plt.show()