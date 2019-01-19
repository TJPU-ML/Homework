import numpy as np
import random

train_set = []
test_set = []
iris = ([], [], [])
map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

data = np.genfromtxt(IRIS_PATH, delimiter=',', usecols=range(5))
target = np.genfromtxt(IRIS_PATH, delimiter=',', usecols=(4), dtype=str)

for i in range(len(data)):
    idx = map[target[i]]
    data[i][4] = idx
    iris[idx].append(data[i])

for i in range(2):  # 取前两种属性
    random.shuffle(iris[i])
    si = int(len(iris[i]) * TT_RATE)
    train_set += iris[i][:si]
    test_set += iris[i][si:]


random.shuffle(train_set)
random.shuffle(test_set)
train_set = np.array(train_set)
test_set = np.array(test_set)

np.savetxt(TRAIN_SET_PATH(), train_set, fmt='%.1f', delimiter=',')
np.savetxt(TEST_SET_PATH(), test_set, fmt='%.1f', delimiter=',')

