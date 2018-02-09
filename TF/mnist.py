from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
	tf.app.run()

def main():
	print(mnist)