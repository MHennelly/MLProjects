import numpy as np

data = np.genfromtxt('housing.csv', dtype=float, delimiter=',')
data_rows = data[1:,1]
weights = np.random.rand(7,1)
features = data[1:1000,0:7]
medianPricing = data[1:1000,8]
step = 0.000001

#Feature Scaling
medianPricing /= 1000
for x in range(8):
    medianPricing = medianPricing
    col = data[1:,x]
    average = np.mean(col)
    fRange = np.max(col) - np.min(col)
    if x < 7:
        features[x] = (features[x] - average) / fRange
    else:
        medianPricing = (medianPricing - average) / fRange

def hypothesis():
    return np.matmul(features, weights)

def cost():
	total = np.sum((np.subtract(hypothesis(),medianPricing))**2)
	return 1 / (2 * data_rows) * total

def gradientDescent(w):
    temp = []
    for x in range(7):
        temp.append(w[x] - step / data_rows * np.sum(np.subtract(hypothesis(),medianPricing))*features[:,x])
    w = temp

print(cost())
