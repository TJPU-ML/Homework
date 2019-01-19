
# coding: utf-8

# In[103]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
iris_path = 'C:/Users/61451/Desktop/iris.csv'
data = pd.read_csv(iris_path)


data = np.array(new_iris) 
     

from sklearn.model_selection import train_test_split
x,y = np.split(data, (4,), axis = 1)  
x_train1, x_test1, y_train1,y_test1 = train_test_split(x,y,test_size = 0.3,random_state = 0)
x_train2, x_test2, y_train2,y_test2 = train_test_split(x,y,test_size = 0.3,random_state = 0)
x_train3, x_test3, y_train3,y_test3 = train_test_split(x,y,test_size = 0.3,random_state = 0)
def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
import random  

theat=[1,1,1,1]      #θ参数初始化  
  
step_size = 0.005
 
max_iters =10000
error =0            #损失值  
iter_count = 0
k=0
j=0
i=0
t=0
erro=[0]*max_iters
for p in range(max_iters):
    cost=[0,0,0,0]
    h=0
    j= random.randint(0,69)        
    h=h+x_train1[j][0]*theat[0]+x_train1[j][1]*theat[1]+x_train1[j][2]*theat[2]+x_train1[j][3]*theat[3]
    error=Sigmoid(h)-y_train1[j]
    erro[p]=error**2
    cost[0]=cost[0]+error*x_train1[j][0]
    cost[1]=cost[1]+error*x_train1[j][1]
    cost[2]=cost[2]+error*x_train1[j][2]
    cost[3]=cost[3]+error*x_train1[j][3]
    theat[0]=theat[0]-cost[0]*step_size
    theat[1]=theat[1]-cost[1]*step_size
    theat[2]=theat[2]-cost[2]*step_size
    theat[3]=theat[3]-cost[3]*step_size
    print(theat[0],theat[1],theat[2],theat[3])
    iter_count += 1  
t=0
f=0
for i in range (30): 
        predy = theat[0]*x_test1[i][0]+theat[1]*x_test1[i][1]+theat[2]*x_test1[i][2]+theat[3]*x_test1[i][3] #预测值  
        h=Sigmoid(predy)
        if h>0.5:
            if y_test1[i]==1:
                t=t+1
            else:
                f=f+1
        elif h<0.5:
            if y_test1[i]==0:
                t=t+1
            else:
                f=f+1
u=t/(t+f)
print(u)
           


# In[97]:


           


# In[98]:


print(erro)


# In[63]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
iris_path = 'C:/Users/61451/Desktop/iris.csv'
data = pd.read_csv(iris_path)
dat = np.array(data)  
dat[0:55,:] 
data.head()

def iris_type(s):
    class_label = {'0':0, '1':1}
    return class_label[s]

new_iris = pd.io.parsers.read_csv(iris_path, converters = {4:iris_type})
new_iris.head()

data = np.array(new_iris)  
data[:10,:]  


# In[ ]:



