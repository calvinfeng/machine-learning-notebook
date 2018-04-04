# Project: Recommender System
# Author(s): Calvin Feng

from csv import reader
from math import sqrt
from pdb import set_trace as debugger

from user import User
from progress import Progress

class IncrementalSVDTester:
    def __init__(self, rating_csv_filepath, trained_movies):
        self.rating_count = 0
        self.test_user_data = dict()

        csv = reader(open(rating_csv_filepath))
        for row in csv:
            if row[0].isdigit():
                self.rating_count += 1
                user_id, movie_id, rating = row[0], row[1], row[2]
                if self.test_user_data.get(user_id):
                    self.test_user_data[user_id]['movie_ratings'][movie_id] = rating
                else:
                    self.test_user_data[user_id] = dict()
                    self.test_user_data[user_id]['movie_ratings'] = { movie_id: rating }

        self.trained_movies = trained_movies

    def configure(self, regularized_factor, learning_rate, latent_factor_length):
        # Define regularization constant to control overfitting
        self.regularized_factor = regularized_factor

        # Define learning rate for gradient descent
        self.learning_rate = learning_rate

        # Define how many latent factors we wish to use for SVD
        self.latent_factor_length = latent_factor_length

        self.users = dict()
        for user_id in self.test_user_data:
            user = self.test_user_data[user_id]
            self.users[user_id] = User(user_id, user['movie_ratings'], latent_factor_length, True)

    @property
    def training_rmse(self):
        sq_error = 0
        m = 0
        for user_id in self.users:
            user = self.users[user_id]
            for movie_id in user.movie_ratings:
                if self.trained_movies.get(movie_id):
                    movie = self.trained_movies[movie_id]
                    sq_error += (movie.hypothesis(user) - float(user.movie_ratings[movie_id]))**2
                    m += 1

        return sqrt(sq_error / m)

    @property
    def test_rmse(self):
        sq_error = 0
        m = 0
        for user_id in self.users:
            user = self.users[user_id]
            for movie_id in user.hidden_ratings:
                if self.trained_movies.get(movie_id):
                    movie = self.trained_movies[movie_id]
                    sq_error += (movie.hypothesis(user) - float(user.hidden_ratings[movie_id]))**2
                    m += 1

        return sqrt(sq_error / m)

    @property
    def content_based_cost(self):
        if self.regularized_factor is None:
            return None

        m = 0

        sq_error = 0
        for user_id in self.users:
            user = self.users[user_id]
            for movie_id in user.movie_ratings:
                if self.trained_movies.get(movie_id):
                    movie = self.trained_movies[movie_id]
                    sq_error += (movie.hypothesis(user) - float(user.movie_ratings[movie.id]))**2
                    m += 1
        sq_error *= (0.5 / m)

        regularized_term = 0
        for user_id in self.users:
            user = self.users[user_id]
            for k in range(0, len(user.theta)):
                regularized_term += user.theta[k]**2
        regularized_term *= ( 0.5 * self.regularized_factor / m)

        return regularized_term + sq_error

    '''
    Partial derivatives, wrt => with respect to
    TODO: Combine the following two into one function
    '''
    def dj_wrt_user_theta_k(self, user, k):
        m = 0
        derivative_sum = 0
        for movie_id in user.movie_ratings:
            if self.trained_movies.get(movie_id):
                movie = self.trained_movies[movie_id]
                derivative_sum += (movie.hypothesis(user) - float(user.movie_ratings[movie_id])) * movie.feature[k]
                m += 1

        if m == 0:
            return derivative_sum + (self.regularized_factor * user.theta[k])

        return (derivative_sum / m) + (self.regularized_factor * user.theta[k] / m)

    def dj_wrt_user_theta_k0(self, user):
        m = 0
        derivative_sum = 0
        for movie_id in user.movie_ratings:
            if self.trained_movies.get(movie_id):
                movie = self.trained_movies[movie_id]
                derivative_sum += (movie.hypothesis(user) - float(user.movie_ratings[movie_id])) * movie.feature[0]
                m += 1

        if m == 0:
            return derivative_sum

        return (derivative_sum / m)

    def content_based_batch_gradient_descent(self):
        if self.learning_rate is None or self.regularized_factor is None:
            return False

        total_iteration = 1000
        progress = Progress('Content-based Gradient Descent', total_iteration)

        log = []
        current_iteration = 1
        while current_iteration <= total_iteration:
            progress.report(current_iteration, self.content_based_cost)

            # ==> Compute partial derivatives
            # Derivative of cost function wrt movie features
            dj_duser = dict()
            for user_id in self.users:
                user = self.users[user_id]
                n = len(user.theta)
                dj_duser[user.id] = []
                for k in range(0, n):
                    if k == 0:
                        dj_duser[user.id].append(self.dj_wrt_user_theta_k0(user))
                    else:
                        dj_duser[user.id].append(self.dj_wrt_user_theta_k(user, k))

            # Apply gradient descent
            for user_id in dj_duser:
                dj_dtheta = dj_duser[user_id]
                user = self.users[user_id]
                n = len(user.theta)
                for k in range(0, n):
                    user.theta[k] = user.theta[k] - (self.learning_rate * dj_dtheta[k])

            current_iteration += 1
        progress.complete()
        return log
