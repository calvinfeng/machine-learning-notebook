import numpy as np
from progress_bar import ProgressBar


class Factorizer(object):
    def __init__(self, training_rating_matrix, test_rating_matrix, reg=0.0, feature_dim=10):
        """Instantiate the necessary matrices to perform low rank factorization

        :param training_rating_matrix numpy.array: A sparse numpy matrix that holds ratings from every user on every movie.
        :param test_rating_matrix numpy.array: A sparse numpy matrix that holds ratings from every user on every movie.
        :param reg float: Regularization strength
        :param learninG_rate float: Learning rate for gradient descent update rule
        :param feature_dim int: Dimension for the latent vector, i.e. feature for movie, preference for user
        """
        if training_rating_matrix.shape != test_rating_matrix.shape:
            raise "dimension of training set does not match up with that of test set"

        self.reg = reg
        self.feature_dim = feature_dim
        self.user_dim, self.movie_dim = training_rating_matrix.shape

        # Instantiate the low rank factorizations
        self.U = np.random.rand(self.user_dim, self.feature_dim)
        self.M = np.random.rand(self.movie_dim, self.feature_dim)
        self.R = training_rating_matrix
        self.R_test = test_rating_matrix
        print "Factorizer is instantiated with U: %s and M: %s" % (self.U.shape, self.M.shape)

    def loss(self):
        """Computes L2 loss
        """
        pred_R = np.dot(self.U, self.M.T)

        loss, rmse, num_test_ratings = 0, 0, 0
        itr = np.nditer(self.R, flags=['multi_index'])  # itr stands for iterator
        while not itr.finished:
            # When computing loss, only consider training data
            if self.R[itr.multi_index] != 0 and self.R_test[itr.multi_index] == 0:
                loss += 0.5 * (self.R[itr.multi_index] - pred_R[itr.multi_index])**2

            # When computing RMSE, only consider test data
            if self.R_test[itr.multi_index] != 0 and self.R[itr.multi_index] == 0:
                rmse += (self.R_test[itr.multi_index] - pred_R[itr.multi_index])**2
                num_test_ratings += 1

            itr.iternext()

        # Factor in regularizations
        loss += self.reg * np.sum(self.U * self.U) / 2
        loss += self.reg * np.sum(self.M * self.M) / 2
        rmse = np.sqrt(rmse / num_test_ratings)

        return loss, rmse

    def gradients(self):
        grad_R = np.dot(self.U, self.M.T) - self.R
        grad_R[self.R == 0] = 0
        grad_u = np.dot(grad_R, self.M) + (self.reg * self.U)
        grad_m = np.dot(grad_R.T, self.U) + (self.reg * self.M)

        return grad_u, grad_m

    def train(self, learning_rate=1e-5, steps=200, epoch=10):
        benchmarks = []
        progress = ProgressBar('training', steps)
        for step in range(steps):
            if step % epoch == 0:
                loss, rmse = self.loss()
                benchmarks.append((step + 1, loss, rmse))
                progress.report(step, loss)
            grad_u, grad_m = self.gradients()
            self.U = self.U - (learning_rate * grad_u)
            self.M = self.M - (learning_rate * grad_m)

        loss, rmse = self.loss()
        benchmarks.append((step + 1, loss, rmse))
        progress.complete()
        return benchmarks

    def num_gradients(self, h=1e-5):
        """Compute numerical gradients for U and M. Please be cautious of this function; it has extremely bad
        time complexity. It is meant for testing purpose.

        :param float h: Small delta for computing the slope at a given point.
        """
        num_grad_u = np.zeros(self.U.shape)
        num_grad_m = np.zeros(self.M.shape)

        U_dim, L_dim = self.U.shape
        M_dim, L_dim = self.M.shape

        itr = np.nditer(num_grad_u, flags=['multi_index'])
        while not itr.finished:
            indices = itr.multi_index

            # Store the old value
            old_val = self.U[indices]

            self.U[indices] = old_val + h
            fuph, _ = self.loss()

            self.U[indices] = old_val - h
            fumh, _ = self.loss()

            self.U[indices] = old_val
            num_grad_u[indices] = (fuph - fumh) / (2 * h)

            itr.iternext()

        itr = np.nditer(num_grad_m, flags=['multi_index'])
        while not itr.finished:
            indices = itr.multi_index

            # Store the old value
            old_val = self.M[indices]

            self.M[indices] = old_val + h
            fmph, _ = self.loss()

            self.M[indices] = old_val - h
            fmmh, _ = self.loss()

            self.M[indices] = old_val
            num_grad_m[indices] = (fmph - fmmh) / (2 * h)

            itr.iternext()

        return num_grad_u, num_grad_m
