import tensorflow as tf
import numpy as np

"""
Two-layer ReLU fully connected network running on Tensorflow with optimized the variables, i.e. we are not going to load
weights from CPU to GPU and GPU back to CPU. The weights will stay within the graph throughout training.
"""

N, D, H = 64, 1000, 100
with tf.device('/cpu:0'):
    # Define our computational graph
    x = tf.placeholder(tf.float32, shape=(N,D))
    y = tf.placeholder(tf.float32, shape=(N,D))

    # Notice that I use Variable instead of placeholder to avoid transfering values from GPU to CPU if
    # we were to run this with GPU
    w1 = tf.Variable(tf.random_normal((D, H)), name="w1")
    w2 = tf.Variable(tf.random_normal((H, D)), name="w2")

    h = tf.maximum(tf.matmul(x, w1), 0)
    y_pred = tf.matmul(h, w2)
    diff = y_pred - y

    # L2 Loss
    loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))

    # Telling TensorFlow the update rules for the tf Variables
    learning_rate = 1e-5
    grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
    new_w1 = w1.assign(w1 - learning_rate * grad_w1)
    new_w2 = w2.assign(w2 - learning_rate * grad_w2)

    # Create a new dummy node to explicitly tell TensorFlow to NOT SKIP computing it
    weight_updates = tf.group(new_w1, new_w2)

with tf.Session() as sess:
    print sess.run(tf.report_uninitialized_variables())
    sess.run(tf.global_variables_initializer())
    # Vectors x and y should not live in the graph as variables because we actually want to load many different
    # mini batch into the graph for SGD.
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D)
    }

    losses = []
    for t in range(50):
        loss_val, _ = sess.run([loss, weight_updates], feed_dict=values)
        losses.append(loss_val)

    print losses
