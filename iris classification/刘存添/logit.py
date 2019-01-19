import time
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    if z >= 0:
        return 1.0/(1+np.exp(-z))
    else:
        return np.exp(z)/(1+np.exp(z))


def coe(B): # 求导
    tmp = np.zeros(5)
    for c in train_set:
        x, y = c[:4], c[-1]
        X = np.append(x, [1])
        w, b = B[:4], B[-1]
        p1 = np.exp(np.dot(w.T, x) + b)
        tmp += np.dot(X, y - p1 / (1 + p1))
    return -tmp


def cost_func(h, y): # 损失函数
    return -y * np.log(h) - (1 - y) * np.log(1-h)


bh = [np.zeros(5)]

train_set = np.genfromtxt(TRAIN_SET_PATH(), delimiter=',')
test_set = np.genfromtxt(TEST_SET_PATH(), delimiter=',')

print('TRAIN : TEST ')
print('%5d : %-5d' % (len(train_set), len(test_set)))

start = time.time()

for i in range(TIMES): # 训练
    b = bh[i]
    bh.append(b - np.dot(LEARN_RATE, coe(b)))

end = time.time()
print('Done. Took %.3f seconds.' % (end - start))

pots = []
for i in range(0, TIMES, 1):
    B = bh[i]
    svm = 0.0
    for c in test_set: # 测试
        x, y = c[:4], c[-1]
        w, b = B[:4], B[-1]
        h = sigmoid(np.dot(w.T, x) + b)
        svm += cost_func(h, y)
    svm /= len(test_set)
    pots.append(svm)

print('正确率：%f%%'%((1-pots[-1])*100))
plt.plot(pots, label='%d%%' % int(TT_RATE*100))

plt.xlabel('Train times')
plt.ylabel('Price')
plt.legend()
plt.savefig('picture/%d-%s-test.png' % (int(TT_RATE*100), TIMES))
