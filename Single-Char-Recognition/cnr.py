import tensorflow as tf
import cnr_input
import matplotlib.pyplot as plt
import numpy as np

x = tf.placeholder(tf.float32, [None, 448])
W = tf.Variable(tf.zeros([448, 31]))
b = tf.Variable(tf.zeros([31]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 31])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_image, train_label = cnr_input.train_inputs()
test_image, test_label = cnr_input.test_inputs()

it = []
train_ac = []
test_ac = []

steps = 2000
for i in range(steps):
    sess.run(train_step, feed_dict={x: train_image, y_: train_label})

    if i % 10 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('step %d, train accuracy = %.3f%%' % (i, np.multiply(
            sess.run(accuracy, feed_dict={x: train_image, y_: train_label}), 100.0)))
        print('step %d, test accuracy = %.3f%%' % (i, np.multiply(
            sess.run(accuracy, feed_dict={x: test_image, y_: test_label}), 100.0)))
        it.append(i)
        train_ac.append(sess.run(accuracy, feed_dict={x: train_image, y_: train_label}))
        test_ac.append(sess.run(accuracy, feed_dict={x: test_image, y_: test_label}))

np.array(sess.run(W)).tofile('cnr_array_w')
np.array(sess.run(b)).tofile('cnr_array_b')

plt.plot(it, train_ac, color='blue', label='train')
plt.plot(it, test_ac, color='red', label='test')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()
