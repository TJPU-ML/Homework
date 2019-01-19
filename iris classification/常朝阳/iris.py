#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入所需要的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected = True)


# In[2]:


data = pd.read_csv('iris.data.csv',header=None) #利用pandas解析csv数据


# In[3]:


data


# In[4]:


newdata = np.array(data)#转为numpy矩阵
newdata


# In[5]:


datas = newdata[:100,:2]#数据取前两类,前两列(即Sepal length与Sepal width)选为属性
datas
datas.shape


# In[14]:


labels = newdata[:100,4]#数据取前两类(即setosa与versicolor),第五列选为标签
labels


# In[7]:


kinds=list(set(labels))#查看种类
kinds


# In[8]:


#数据的二维可视化散点图
X = [x[0] for x in datas]  
 
Y = [x[1] for x in datas]  

plt.scatter(X[:50], Y[:50], color='red', marker='o', label='setosa') 
plt.scatter(X[50:100], Y[50:100], color='blue', marker='x', label='versicolor') 

plt.legend(loc=2) 
plt.show()


# In[9]:


X = datas
Y = labels
#利用train_test_split函数,按照7:3比例将数据分为训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)


# In[10]:


#画出训练集数据点
trace = go.Scatter(x = X[:,0], y = X[:,1], mode = 'markers',marker = dict(color = np.random.randn(100),size = 10, showscale=False))
layout = go.Layout(xaxis=dict(title='Sepal length', showgrid=False),
                    yaxis=dict(title='Sepal width',showgrid=False),
                    width = 700, height = 380)
fig = go.Figure(data=[trace], layout=layout)


# In[11]:


iplot(fig)


# In[12]:


lr = LogisticRegression(penalty='l2',solver='newton-cg',multi_class='multinomial')
lr.fit(x_train,y_train)


# In[13]:


y_hat = lr.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_hat)
print("对数几率回归模型正确率：%.2f" %accuracy)

