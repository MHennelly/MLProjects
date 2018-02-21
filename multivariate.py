import numpy as np

data = np.genfromtxt('housing.csv', dtype=float, delimiter=',')
data_rows = len(data[1:,1])
features = data[1:,1:7]
feature_cols = len(features[1,:])
medianPricing = data[1:,8]
weights = np.random.rand(feature_cols)
step = 0.000000129

#Feature Scaling
"""medianPricing /= 1000
for x in range(8):
    medianPricing = medianPricing
    col = data[1:,x]
    average = np.mean(col)
    fRange = np.max(col) - np.min(col)
    if x < 7:
        features[x] = (features[x] - average) / fRange
    else:
        medianPricing = (medianPricing - average) / fRange"""

def hypothesis():
    return np.matmul(features, weights)

def cost():
	return 1 / (2 * data_rows) * (np.sum(np.subtract(hypothesis(),medianPricing).astype(int)**2))

def gradientDescent(w):
    temp = []
    for x in range(feature_cols):
        y = np.subtract(hypothesis(),medianPricing)
        z = np.multiply(y,features[:,x])
        temp.append(w[x] - step / data_rows * np.nansum(z))
    return temp

print("Original Cost: ",cost())
print("Original Weights: ",weights)
x = hypothesis()
print("Example Original Hypotheses: ",x[0:10])
print("Real Value: ",medianPricing[0:10])
print("Now Performing Gradient Descent...")
for n in range(10000):
    weights = gradientDescent(weights)
print("Final Cost: ",cost())
print("Final Weights: ",weights)
x = hypothesis()
print("Example Final Hypotheses: ",x[0:10])
print("Real Value: ",medianPricing[0:10])
