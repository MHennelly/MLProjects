import numpy as np

data = np.genfromtxt('cars.csv', dtype=int, delimiter=',')
data_rows = 50
weights = np.random.rand(2,1)
speed = data[1:,1]
distance = data[1:,2]
step = 0.01

def hypothesis():
	return weights[0] + weights[1] * speed

def cost():
	total = np.sum((hypothesis() - distance)**2)
	return 1 / (2 * data_rows) * total

def lms():
	temp0 = weights[0] - step * np.sum(hypothesis() - distance) / data_rows
	temp1 = weights[1] - step * np.sum(np.multiply(hypothesis() - distance, speed)) / data_rows
	weights[0] = temp0
	weights[1] = temp1

print(hypothesis())

for x in range(10):
	print(cost())
	lms()
