import numpy as np

data = np.genfromtxt('iris.txt', delimiter=',', usecols = (0,1,2,3))
labels = np.genfromtxt('iris.txt',delimiter=',', dtype=str,usecols=4)
feature_cols = len(data[1,:])
data_size = len(data[:,1])
wSet = np.random.rand(feature_cols)
wVer = np.random.rand(feature_cols)
wVirg = np.random.rand(feature_cols)

def labelToData(name):
    X = np.zeros((data_size,1))
    for i in range(data_size):
        if labels[i] == name:
            X[i] = 1
        else:
            X[i] = 0
    return X

dSet = labelToData('Iris-setosa')
dVer = labelToData('Iris-versicolor')
dVirg = labelToData('Iris-virginica')

def hSet():
    z = np.matmul(data,wSet)
    return 1 / (1 + np.exp(-z))
def hVer():
    z = np.matmul(data,wVer)
    return 1 / (1 + np.exp(-z))
def hVirg():
    z = np.matmul(data,wVirg)
    return 1 / (1 + np.exp(-z))

def costSet():
    H = hSet()
    m1 = np.matmul(np.transpose(dSet),np.log(H))
    m2 = np.matmul(np.transpose(-dSet + 1),np.log(-H+1))
    return -1 / data_size * np.sum(m1+m2)

def costVer():
    H = hSet()
    m1 = np.matmul(np.transpose(dVer),np.log(H))
    m2 = np.matmul(np.transpose(-dVer + 1),np.log(-H+1))
    return -1 / data_size * np.sum(m1+m2)

def costVirg():
    H = hSet()
    m1 = np.matmul(np.transpose(dVirg),np.log(H))
    m2 = np.matmul(np.transpose(-dVirg + 1),np.log(-H+1))
    return -1 / data_size * np.sum(m1+m2)

print(hSet())
print(costSet())
print(costVer())
print(costVirg())
