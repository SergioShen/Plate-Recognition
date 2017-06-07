import tensorflow as tf
import scr_input
import matplotlib.pyplot as plt

train_image, train_label = scr_input.train_inputs()
test_image, test_label = scr_input.test_inputs()

x = tf.placeholder(tf.float32, [None, 288])
x_image = tf.reshape(x, [-1, 12, 24, 1])
y_ = tf.placeholder(tf.float32, [None, 34])
sess = tf.InteractiveSession()

# cnn below:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# build a 1->16->24->256->34 cnn
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([3 * 6 * 64, 512])
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 6 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([512, 34])
b_fc2 = bias_variable([34])

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

# training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

it = []
train_acc = []
test_acc = []

for i in range(10000):
    if i % 25 == 0:
        print("")
        train_accuracy = accuracy.eval(feed_dict={x: train_image, y_: train_label, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x: test_image, y_: test_label, keep_prob: 1.0})
        print("step %d, training accuracy %g%%, test accuracy %g%%" % (i, 100 * train_accuracy, 100 * test_accuracy))
        it.append(i)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
    print('.', end="", flush=True)
    train_step.run(feed_dict={x: train_image, y_: train_label, keep_prob: 0.5})
print("Final test accuracy %g%%" % 100 * accuracy.eval(feed_dict={x: test_image, y_: test_label, keep_prob: 1.0}))

saver = tf.train.Saver()
save_path = saver.save(sess, 'scr_cn.data')

plt.plot(it, train_acc, color='green', label='train')
plt.plot(it, test_acc, color='red', label='test')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()
