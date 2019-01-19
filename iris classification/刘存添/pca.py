import numpy as np
import matplotlib.pyplot as plt

iris_path = 'iris.data'
map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

x = np.genfromtxt(iris_path, delimiter=',', usecols=range(4))

mean_x = x.mean(axis=0)
x = x - mean_x  # 去中心化
cov_x = np.cov(x, rowvar=0)  # 协方差矩阵
eval, evec = np.linalg.eig(np.mat(cov_x))  # 特征值与特征向量
idx = np.argsort(eval)
idx = idx[:-3:-1]
x = x * evec[:, idx]


plt.scatter(x[:50][:, 0].tolist(), x[:50][:, 1].tolist(),
            c='red', label='Iris-setosa')
plt.scatter(x[50:100][:, 0].tolist(), x[50:100]
            [:, 1].tolist(), c='blue', label='Iris-versicolor')
plt.scatter(x[100:][:, 0].tolist(), x[100:][:, 1].tolist(),
            c='green', label='Iris-virginica')
plt.legend()
plt.savefig('picture/pca.png')
plt.show()
