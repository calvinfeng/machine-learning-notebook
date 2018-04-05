import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import math


def simple_model(X, y):
    # Define variables
    Wconv1 = tf.get_variable('Wconv1', shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable('bconv1', shape=[32])
    W1 = tf.get_variable('W1', shape=[5408, 10]) # 5408 is the result of first convolution, 32 * 13 * 13
    b1 = tf.get_variable('b1', shape=[10])

    # Define the computational graph
    conv_out = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding='VALID') + bconv1
    relu_out = tf.nn.relu(conv_out)
    relu_out_flat = tf.reshape(relu_out, [-1, 5408]) # -1 is used to infer the shape
    pred = tf.matmul(relu_out_flat, W1) + b1

    return pred

def run_model(session, pred, X, y, is_training, loss_val, Xdata, ydata, epochs=1, batch_size=64, print_every=100, train_step=None, plot_losses=False):
    # Compute accuracy using tf
    correct_prediction = tf.equal(tf.argmax(pred, axis=1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Shuffle indices
    N = Xdata.shape[0]
    train_indices = np.arange(N)
    np.random.shuffle(train_indices)

    is_training_mode = train_step is not None

    # Set up variables for computation and optimization
    variables = [mean_loss, correct_prediction, accuracy]
    if is_training_mode:
        variables[-1] = train_step

    iter_counter = 0
    for e in range(epochs):
        # Keep track of losses and accuracy
        num_correct = 0
        losses = []
        for i in range(int(math.ceil(N / batch_size))):
            # Generate indices for the batch
            start_idx = (i * batch_size) % N
            idx = train_indices[start_idx:start_idx + batch_size]

            feed_dict = {
                X: Xdata[idx, :],
                y: ydata[idx],
                is_training: is_training_mode
            }

            actual_batch_size = ydata[idx].shape[0]

            # Compute loss and number of correct predictions
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            losses.append(loss * actual_batch_size)
            num_correct += float(np.sum(corr))

            if is_training_mode and (iter_counter % print_every) == 0:
                print "Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(
                    iter_counter, loss, float(np.sum(corr)) / actual_batch_size)

            iter_counter += 1

        accuracy = num_correct / N
        avg_loss = np.sum(losses) / N
        print "Epoch {0}, overall loss = {1:.3g} and accuracy = {2:.3g}".format(e+1, avg_loss, accuracy)

        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return avg_loss, accuracy


if __name__ == '__main__':
    tf.reset_default_graph()

    # Setup input, e.g. data that changes every batch
    X = tf.placeholder(tf.float32, [None, 32, 32, 3]) # First dim is None, and gets set automatically based on batch size
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    # Construct model
    tf_pred = simple_model(X, y)

    # Define loss
    total_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=tf_pred)
    mean_loss = tf.reduce_mean(total_loss)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(5e-4) # Set learning rate
    train_step = optimizer.minimize(mean_loss)

    data = data_utils.get_preprocessed_CIFAR10('datasets/cifar-10-batches-py', should_transpose=False)
    Xtr, ytr = data['X_train'], data['y_train']

    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            sess.run(tf.global_variables_initializer())
            run_model(sess, tf_pred, X, y, is_training, mean_loss, Xtr, ytr,
                    epochs=100, batch_size=64, print_every=100, train_step=train_step, plot_losses=False)
