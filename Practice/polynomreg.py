import numpy as np

data = np.genfromtxt('cars.csv', dtype=int, delimiter=',')
data_rows = 50
weights = np.random.rand(3,1)
speed = data[1:,1]
distance = data[1:,2]
step = 0.00038

def hypothesis():
	return weights[0] + weights[1] * speed + weights[2] * speed ** 2
	#return weights[2] * speed ** 2

def cost():
	total = np.sum((hypothesis() - distance)**2)
	return 1 / (2 * data_rows) * total

def lms():
	temp0 = weights[0] - step * np.sum(hypothesis() - distance) / data_rows
	temp1 = weights[1] - step * np.sum(np.multiply(hypothesis() - distance, speed)) / data_rows
	temp2 = weights[2] - step * np.sum(np.multiply(hypothesis() - distance, speed)) / data_rows
	weights[0] = temp0
	weights[1] = temp1
	weights[2] = temp2

print("Initial Hypothesis: ", hypothesis())
print("Cost: " + str(cost()))
print("Initial Weights: ", weights)

for x in range(100000):
	lms()

print("Final Hypothesis: ", hypothesis())
print("Actual Values: ", distance)
print("Cost: " + str(cost()))
print("Final Weights: ", weights)
