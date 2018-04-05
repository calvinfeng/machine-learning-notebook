import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import math
import time


class MediumModel(object):
    """A medmium complexity model

    - 7x7 convolutional layer with 32 filters, stride = 1
    - ReLU
    - Spatial batch normalization
    - 2x2 max pooling, stride = 2
    - Affine with weight of shape (13*13*32, 1024)
    - ReLU
    - Affine with weight of shape (1024, 10)
    """
    def __init__(self):
        tf.reset_default_graph()

        # Define placeholders
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3]) # First dim is None, and gets set automatically based on batch size
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)

        # Define variables
        W1 = tf.get_variable('W1', shape=[7, 7, 3, 32]) # Conv filter weights
        b1 = tf.get_variable('b1', shape=[32]) # Conv filter bias

        # H_out, W_out = (13, 13), output will be (N, 32, 13, 13), and reshape to (N, 32*13*13)

        W2 = tf.get_variable('W2', shape=[5408, 1024]) # 5408 is the result of first convolution, 32 * 13 * 13
        b2 = tf.get_variable('b2', shape=[1024])

        W3 = tf.get_variable('W3', shape=[1024, 10])
        b3 = tf.get_variable('b3', shape=[10])

        # Define the computational graph
        conv_out_1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='VALID') + b1
        relu_out_2 = tf.nn.relu(conv_out_1)
        bn_out_3 = tf.layers.batch_normalization(relu_out_2, axis=3, training=self.is_training)
        pool_out_4 = tf.nn.pool(bn_out_3, window_shape=[2, 2], pooling_type='MAX', padding='VALID', strides=[2, 2])
        pool_out_4 = tf.reshape(pool_out_4, [-1, 32*13*13]) # -1 is used to infer N
        affine_out_5 = tf.matmul(pool_out_4, W2) + b2
        relu_out_6 = tf.nn.relu(affine_out_5)
        affine_out_7 = tf.matmul(relu_out_6, W3) + b3

        # Define SVM loss
        hinge_loss = tf.losses.hinge_loss(labels=tf.one_hot(self.y, 10), logits=affine_out_7)
        self.mean_loss = tf.reduce_mean(hinge_loss)

        # Define prediction and accuracy
        self.correct_prediction = tf.equal(tf.argmax(affine_out_7, axis=1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Define optimization objective, a.k.a train step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensure that we execute the update_ops before performing the optimization step
            self.op_objective = tf.train.AdamOptimizer(5e-4).minimize(self.mean_loss)

    def run(self, inputX, inputY, epochs=1, print_every=70, batch_size=100, mode='train'):
        iterations = []
        minibatch_losses = []
        minibatch_accurs = []
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                sess.run(tf.global_variables_initializer())

                N = inputX.shape[0]
                train_indices = np.arange(N)
                np.random.shuffle(train_indices)

                variables = [self.mean_loss, self.correct_prediction, self.op_objective]
                if mode != 'train':
                    # If testing, we compute an accuracy score instead of the optimization objective
                    variables[-1] = self.accuracy

                iter_cnt = 0
                for ep in range(epochs):
                    num_correct_per_epoch = 0
                    num_iterations = int(math.ceil(N / batch_size))
                    for i in range(num_iterations):
                        # Generate indices for the batch
                        start_idx = (i * batch_size) % N
                        idx_range = train_indices[start_idx:start_idx + batch_size]

                        feed_dict = {
                            self.X: inputX[idx_range, :],
                            self.y: inputY[idx_range],
                            self.is_training: mode == 'train'
                        }

                        actual_batch_size = inputY[idx_range].shape[0]

                        # Compute loss and number of correct predictions
                        mean_loss, corr, _ = sess.run(variables, feed_dict=feed_dict)

                        minibatch_accurs.append(float(np.sum(corr)) / actual_batch_size)
                        minibatch_losses.append(mean_loss * actual_batch_size)
                        iterations.append(iter_cnt)

                        num_correct_per_epoch += float(np.sum(corr))

                        if mode == 'train' and (iter_cnt % print_every) == 0:
                            mini_batch_acc = float(np.sum(corr)) / actual_batch_size
                            mini_batch_loss = mean_loss * actual_batch_size
                            print "Iteration {0}: with mini-batch loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt, mini_batch_loss, mini_batch_acc)

                        iter_cnt += 1

                    # End of epoch, that means went over all training examples at least once.
                    accuracy = num_correct_per_epoch / N
                    avg_loss = np.sum(minibatch_losses) / N
                    print "Epoch {0}, overall loss = {1:.3g} and training accuracy = {2:.3g}".format(ep+1, avg_loss, accuracy)

        return iterations, minibatch_losses, minibatch_accurs


if __name__ == '__main__':
    model = MediumModel()
    data = data_utils.get_preprocessed_CIFAR10('datasets/cifar-10-batches-py', should_transpose=False)
    Xtr, ytr = data['X_train'], data['y_train']
    print "Training data X shape={0}".format(Xtr.shape)
    print "Training data y shape={0}".format(ytr.shape)

    t0 = time.time()
    iterations, minibatch_losses, minibatch_accurs = model.run(Xtr, ytr,
                                                                epochs=6,
                                                                batch_size=100,
                                                                print_every=70)
    t1 = time.time()

    print "Elapsed time using GPU: " + str(t1 - t0)

    plt.grid(True)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(iterations, minibatch_losses)
    plt.title('Mini-batch Loss')
    plt.xlabel('Iteration number')
    plt.ylabel('Mini-batch training loss')

    plt.figure(1)
    plt.subplot(2, 1, 2)
    plt.plot(iterations, minibatch_accurs)
    plt.title('Training Accuracy')
    plt.xlabel('Iteration number')
    plt.ylabel('Mini-batch training accuracy')
    plt.show()

    Xval, yval = data['X_val'], data['y_val']
    iterations, minibatch_losses, minibatch_accurs = model.run(Xval, yval, epochs=1, batch_size=100, mode='validate')
    print minibatch_accurs
