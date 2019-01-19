'''
没有Python灵魂的程序，写法实在太笨重了
'''

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

data_need = load_iris()       #获取原始数据  
'''
分割数据：
sl-->sepal length
sw-->sepal width
pl-->petal length
pw-->petal width
'''                                                        
sl = data_need.data[:50,:1]
sl1 = data_need.data[50:100,:1]
sl2 = data_need.data[100:,:1]

sw = data_need.data[:50,1:2]
sw1 = data_need.data[50:100,1:2]
sw2 = data_need.data[100:,1:2]

pl = data_need.data[:50,2:3]
pl1 = data_need.data[50:100,2:3]
pl2 = data_need.data[100:,2:3]

pw = data_need.data[:50,-1]
pw1 = data_need.data[50:100,-1]
pw2 = data_need.data[100:,-1]

'''
可视化散点图
三行两列
'''
plt.figure(figsize=(15, 15))
plt.subplot(3,2,1)
plt.scatter(sw,sl,color = 'blue',marker = 'o',label = 'Setosa ')
plt.scatter(sw1,sl2,color = 'green',marker = '+',label = 'Versicolour')
plt.scatter(sw2,sl2,color = 'red',marker = 'D',label = 'Virginica')
plt.xlabel('sepal width')
plt.ylabel('sepal length')
plt.legend(loc = 'upper right')

plt.subplot(3,2,2)
plt.scatter(pl,sw,color = 'blue',marker = 'o',label = 'Setosa ')
plt.scatter(pl1,sw2,color = 'green',marker = '+',label = 'Versicolour')
plt.scatter(pl2,sw2,color = 'red',marker = 'D',label = 'Virginica')
plt.xlabel('petal length')
plt.ylabel('sepal width')
plt.legend(loc = 'upper right')

plt.subplot(3,2,3)
plt.scatter(pl,sl,color = 'blue',marker = 'o',label = 'Setosa ')
plt.scatter(pl1,sl1,color = 'green',marker = '+',label = 'Versicolour')
plt.scatter(pl2,sl2,color = 'red',marker = 'D',label = 'Virginica')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper right')

plt.subplot(3,2,4)
plt.scatter(pw,sw,color = 'blue',marker = 'o',label = 'Setosa ')
plt.scatter(pw1,sw2,color = 'green',marker = '+',label = 'Versicolour')
plt.scatter(pw2,sw2,color = 'red',marker = 'D',label = 'Virginica')
plt.xlabel('petal width')
plt.ylabel('sepal width')
plt.legend(loc = 'upper right')

plt.subplot(3,2,5)
plt.scatter(pw,sl,color = 'blue',marker = 'o',label = 'Setosa ')
plt.scatter(pw1,sl2,color = 'green',marker = '+',label = 'Versicolour')
plt.scatter(pw2,sl2,color = 'red',marker = 'D',label = 'Virginica')
plt.xlabel('petal width')
plt.ylabel('sepal length')
plt.legend(loc = 'upper right')

plt.subplot(3,2,6)
plt.scatter(pw,pl,color = 'blue',marker = 'o',label = 'Setosa ')
plt.scatter(pw1,pl2,color = 'green',marker = '+',label = 'Versicolour')
plt.scatter(pw2,pl2,color = 'red',marker = 'D',label = 'Virginica')
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.legend(loc = 'upper right')

savefig("E:\document\Python\ML\iris_picture.jpg")
plt.show()
