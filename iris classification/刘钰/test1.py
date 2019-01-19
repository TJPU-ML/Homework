from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#将特征换为01
def iris(s):
    class_label={b'setosa':0,b'versicolor':1}
    return class_label[s]
#数据读取
filepath='C:/Users/TAOris/Desktop/iris.txt'
data=np.loadtxt(filepath,dtype=float,converters={4:iris})
#划分训练集和测试集 用split切片
X ,y=np.split(data,(4,),axis=1)
x=X[:,0:2]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=0,test_size=0.5)

#建模
classifier=Pipeline([('sc',StandardScaler()),('clf',LogisticRegression())])

classifier.fit(x_train,y_train.ravel())
def show_accuracy(y_hat,y_test,parameter):
     pass

#计算准确率
print("训练集的准确率为：",classifier.score(x_train,y_train))
y_hat=classifier.predict(x_train)
show_accuracy(y_hat,y_train,'训练集')
print("测试集的准确率为：",classifier.score(x_test,y_test))
y_hat=classifier.predict(x_test)
show_accuracy(y_hat,y_test,'测试集')

print('decision_function:\n', classifier.decision_function(x_train))#距离各类的距离
print('\npredict:\n', classifier.predict(x_train))
# 制图
x1_min = x[:, 0].min()
x1_max = x[:, 0].max()
x2_min= x[:, 1].min()
x2_max = x[:, 1].max()
x1,x2 = np.mgrid[x1_min:x1_max:100j, x2_min:x2_max:100j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

grid_hat = classifier.predict(grid_test)       # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)
# 2.指定默认字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 3.绘制
cm_light = mpl.colors.ListedColormap(['#A0FFA0',  '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'b'])
alpha=0.5
plt.pcolormesh(x1, x2, grid_hat, cmap = cm_light)
plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花分类结果', fontsize=15)
plt.grid()
plt.show()