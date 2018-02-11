import numpy as np

data = np.genfromtxt('housing.csv', dtype=float, delimiter=',')
data_rows = data[1:,1]
weights = np.random.rand(7,1)
features = data[1:,0:7]
medianPricing = data[1:,8]
step = 0.01

def hypothesis():
    return np.matmul(features, weights)

def cost():
	total = np.sum((hypothesis() - medianPricing)**2)
	return 1 / (2 * data_rows) * total

def gradientDescent():
    temp = []
    for x in range(7):
        temp.append(weights[x] - step / data_rows * np.sum((hypothesis() - medianPricing)*features[:,x]))
    weights = temp
