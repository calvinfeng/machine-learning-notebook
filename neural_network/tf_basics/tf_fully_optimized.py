import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from timeit import default_timer

device_name, N, D, H = '/gpu:0', 1000, 1000, 1000

with tf.device(device_name):
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))

    xavier_init = tf.contrib.layers.xavier_initializer()
    h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=xavier_init)
    y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=xavier_init)
    loss = tf.losses.mean_squared_error(y_pred, y)

    optimizer = tf.train.GradientDescentOptimizer(7e0)
    weight_updates = optimizer.minimize(loss)

start = default_timer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D)
    }

    losses, iters = [], []
    for t in range(500):
        loss_val, _ = sess.run([loss, weight_updates], feed_dict=values)
        losses.append(loss_val)
        iters.append(t)

duration = default_timer() - start
print 'Algorithm took %s using %s' % (duration, device_name)
plt.plot(iters, losses)
plt.show()
