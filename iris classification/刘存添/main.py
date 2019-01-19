IRIS_PATH = 'iris.data'
LEARN_RATE = 0.0014
TIMES = 300

def TEST_SET_PATH():
    return 'archive/train-%d.data'%int(TT_RATE*100)
def TRAIN_SET_PATH():
    return 'archive/test-%d.data'%int(TT_RATE*100)

TT_RATE = 0.5
exec(open('iris.py').read()) # 生成测试集 训练集
exec(open('logit.py').read())

TT_RATE = 0.7
exec(open('iris.py').read())
exec(open('logit.py').read())

TT_RATE = 0.9
exec(open('iris.py').read())
exec(open('logit.py').read())