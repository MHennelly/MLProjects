import numpy as np

data = np.genfromtxt('cars.csv', dtype=int, delimiter=',')
data_row = 50
weights = np.random.rand(data_row+1,1)
speed = data[1:,1]
distance = data[1:,2]
step = 0.01

def hypothesis(x):
	return np.matmul(speed[x],weights[x]) + weights[0]

def cost():
	total = 0
	for x in range(data_row):
		total += (speed[x] * weights[x+1] - distance[x])**2
	return 0.5 * total

def lms():
	for j in range(data_row+1):
		weights[j+1] = weights[j+1] - step * (distance[j] - hypothesis(j)) * speed[j]

print(hypothesis())
print(cost())