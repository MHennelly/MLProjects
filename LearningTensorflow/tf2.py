import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
#Number of examples to use for each gradient step
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

#Image of 784 pixels unrolled into one vector 784 elements long
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))

#True label for image
y_true = tf.placeholder(tf.float32, [None, 10])
#Predicted label for image
y_pred = tf.matmul(x, W)

#Error function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

#Gradient descent function
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#The amount of correctly predicted images out of total
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
#Decimal percentage of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:

    #Initialize variables
    sess.run(tf.global_variables_initializer())

    #Train
    for all in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict = {x: batch_xs, y_true: batch_ys})

    #Test our neural network
    ans = sess.run(accuracy, feed_dict = {x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))
