import numpy as np
import random

data = np.genfromtxt('iris.txt', delimiter=',', usecols = (0,1,2,3))
labels = np.genfromtxt('iris.txt',delimiter=',', dtype=str,usecols=4)
feature_cols = len(data[1,:])
data_size = len(data[:,1])
wSet = np.zeros(feature_cols)
wVer = np.random.rand(feature_cols)
wVirg = np.random.rand(feature_cols)
step = 0.001

def labelToData(name):
    X = np.zeros((data_size))
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





def gradSet():
    a = np.transpose(data)
    b = np.subtract(hSet(),dSet)
    return wSet - step / data_size * np.matmul(a,b)
def gradVer():
    a = np.transpose(data)
    b = np.subtract(hVer(),dVer)
    return wVer - step / data_size * np.matmul(a,b)
def gradVirg():
    a = np.transpose(data)
    b = np.subtract(hVirg(),dVirg)
    return wVirg - step / data_size * np.matmul(a,b)



x = random.randint(0,149)
print("Initial Predictions of Random Flower...")
print("Flower: ",labels[x])
temp = hSet()
p1 = temp[x] * 100
temp = hVer()
p2 = temp[x] * 100
temp = hVirg()
p3 = temp[x] * 100
print("Possibility Iris-Setosa: ",p1)
print("Possibility Iris-Versicolor: ",p2)
print("Possibility Iris-Virginica: ",p3)
print("Now Performing Gradient Descent...")
print("First: Iris-Setosa...")
for i in range(10000):
    wSet = gradSet()
print("Second: Iris-Versicolor...")
for i in range(10000):
    wVer = gradVer()
print("Third: Iris-Virginica...")
for i in range(10000):
    wVirg = gradVirg()
print("Final Predictions of Same Flower...")
print("Flower: ",labels[x])
temp = hSet()
p1 = temp[x] * 100
temp = hVer()
p2 = temp[x] * 100
temp = hVirg()
p3 = temp[x] * 100
print("Possibility Iris-Setosa: ",p1)
print("Possibility Iris-Versicolor: ",p2)
print("Possibility Iris-Virginica: ",p3)
total_right = 0
for j in range(1000):
    x = random.randint(0,149)
    temp = hSet()
    p1 = temp[x]
    temp = hVer()
    p2 = temp[x]
    temp = hVirg()
    p3 = temp[x]
    if (p1 > p2 and p1 > p3):
        if x < 50:
            total_right += 1
    elif (p2 > p3):
        if x >= 50 and x < 100:
            total_right += 1
    else:
        if x >= 100:
            total_right += 1
print("Classifier Accuracy From 1000 Trials: ",total_right/1000*100)
