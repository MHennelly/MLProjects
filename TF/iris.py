import tensorflow as tf
import numpy as np

data = np.loadtxt("iris.txt", delimiter=',',dtype=float,usecols=(0,1,2,3))
labels = np.loadtxt("iris.txt",delimiter=',',dtype=str,usecols=(4))
print(data)
print(labels)

num_inputs = 4
num_labels = 3


input_layer = tf.placeholder("float", [None, num_inputs])
weight1 = tf.Variable(tf.random_normal([num_inputs, 1]))
weight2 = tf.Variable(tf.random_normal([num_inputs, 1]))
weight3 = tf.Variable(tf.random_normal([num_inputs, 1]))

def make_nn():
	h1 = tf.matmul(data, weight1)
	h2 = tf.matmul(h1, weight2)
	output = tf.matmul(h2, weight3)
	return output

test = make_nn()

