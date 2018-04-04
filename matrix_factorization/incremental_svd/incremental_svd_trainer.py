# Project: Recommender System
# Author(s): Calvin Feng

from movie import Movie
from user import User
from data_reducer import DataReducer
from progress import Progress

from pdb import set_trace as debugger
from math import sqrt
from csv import writer

class IncrementalSVDTrainer:
    def __init__(self, movie_csv_filepath, rating_csv_filepath, link_csv_filepath):
        self.reducer = DataReducer(movie_csv_filepath, rating_csv_filepath, link_csv_filepath)
        self.regularized_factor = None
        self.learning_rate = None
        self.latent_factor_length = None

    def configure(self, regularized_factor, learning_rate, latent_factor_length):
        # Define regularization constant to control overfitting
        self.regularized_factor = regularized_factor

        # Define learning rate for gradient descent
        self.learning_rate = learning_rate

        # Define how many latent factors we wish to use for SVD
        self.latent_factor_length = latent_factor_length

        self.movies = dict()
        for movie_id in self.reducer.movies:
            movie = self.reducer.movies[movie_id]
            self.movies[movie_id] = Movie(movie_id, movie['title'], movie['user_ratings'], latent_factor_length)

        self.users = dict()
        for user_id in self.reducer.users:
            user = self.reducer.users[user_id]
            self.users[user_id] = User(user_id, user['movie_ratings'], latent_factor_length, True)

    # The cost property represents the value of cost/loss function
    @property
    def cost(self):
        if self.regularized_factor is None:
            return None

        # m denotes number of training examples
        m = 0

        sq_error = 0
        for movie_id in self.movies:
            movie = self.movies[movie_id]
            for user_id in movie.viewer_ids:
                user = self.users[user_id]
                if user.movie_ratings.get(movie.id):
                    sq_error += (movie.hypothesis(user) - float(user.movie_ratings[movie.id]))**2
                    m += 1
        sq_error *= (0.5 / m)

        regularized_term = 0
        for movie_id in self.movies:
            movie = self.movies[movie_id]
            for k in range(0, len(movie.feature)):
                regularized_term += movie.feature[k]**2

        for user_id in self.users:
            user = self.users[user_id]
            for k in range(0, len(user.theta)):
                regularized_term += user.theta[k]**2
        regularized_term *= (0.5 * self.regularized_factor / m)

        return regularized_term + sq_error

    # RMSE stands for Root-mean-squared-error
    @property
    def training_rmse(self):
        sq_error = 0
        m = 0
        for user_id in self.users:
            user = self.users[user_id]
            for movie_id in user.movie_ratings:
                movie = self.movies[movie_id]
                sq_error += (movie.hypothesis(user) - float(user.movie_ratings[movie_id]))**2
                m += 1

        return sqrt(sq_error / m)

    @property
    def cross_validation_rmse(self):
        sq_error = 0
        m = 0
        for user_id in self.users:
            user = self.users[user_id]
            for movie_id in user.hidden_ratings:
                movie = self.movies[movie_id]
                sq_error += (movie.hypothesis(user) - float(user.hidden_ratings[movie_id]))**2
                m += 1

        return sqrt(sq_error / m)

    '''
    Partial derivatives, wrt => with respect to
    '''
    def dj_wrt_movie_feature_k(self, movie, k):
        derivative_sum = 0
        m = 0
        for user_id in movie.viewer_ids:
            user = self.users[user_id]
            if user.movie_ratings.get(movie.id):
                derivative_sum += (movie.hypothesis(user) - float(user.movie_ratings[movie.id])) * user.theta[k]
                m += 1

        if m == 0:
            return derivative_sum + (self.regularized_factor * movie.feature[k])

        return (derivative_sum / m) + (self.regularized_factor * movie.feature[k] / m)

    def dj_wrt_user_theta_k(self, user, k):
        derivative_sum = 0
        m = 0
        for movie_id in user.movie_ratings:
            movie = self.movies[movie_id]
            if movie.user_ratings.get(user.id):
                derivative_sum += (movie.hypothesis(user) - float(user.movie_ratings[movie_id])) * movie.feature[k]
                m += 1

        if m == 0:
            return derivative_sum + (self.regularized_factor * user.theta[k])

        return (derivative_sum / m) + (self.regularized_factor * user.theta[k] / m)


    def batch_gradient_descent(self):
        if self.learning_rate is None or self.regularized_factor is None:
            return False

        total_iteration = 1500
        progress = Progress('Gradient Descent', total_iteration)

        log = []
        current_iteration = 1
        while current_iteration <= total_iteration:
            progress.report(current_iteration, self.cost)

            # ==> Compute partial derivatives
            # Derivative of cost function wrt movie features
            dj_dmovies = dict()
            for movie_id in self.movies:
                movie = self.movies[movie_id]
                n = len(movie.feature)
                dj_dmovies[movie.id] = []
                for k in range(0, n):
                    dj_dmovies[movie.id].append(self.dj_wrt_movie_feature_k(movie, k))

            # Derivative of cost function wrt user preferences
            dj_dusers = dict()
            for user_id in self.users:
                user = self.users[user_id]
                n = len(user.theta)
                dj_dusers[user.id] = []
                for k in range(0, n):
                    dj_dusers[user.id].append(self.dj_wrt_user_theta_k(user, k))

            # Apply gradient_descent
            for movie_id in dj_dmovies:
                dj_dfeature = dj_dmovies[movie_id]
                movie = self.movies[movie_id]
                n = len(movie.feature)
                for k in range(0, n):
                    movie.feature[k] = movie.feature[k] - (self.learning_rate * dj_dfeature[k])

            for user_id in dj_dusers:
                dj_dtheta = dj_dusers[user_id]
                user = self.users[user_id]
                n = len(user.theta)
                for k in range(0, n):
                    user.theta[k] = user.theta[k] - (self.learning_rate * dj_dtheta[k])

            current_iteration +=1
        progress.complete()
        return log

    def export_feature(self, dir):
        feature_length = self.latent_factor_length
        with open(dir + '/movie_features.csv', 'wt') as outfile:
            output = writer(outfile)

            header = ['movieId']
            for k in range(0, feature_length):
                header.append('f%s' % k)
            output.writerow(header)

            for movie_id in self.movies:
                movie = self.movies[movie_id]
                output.writerow([movie.id] + movie.feature)
        return True

if __name__ == '__main__':
    svd = IncrementalSVDTrainer(
                    '../data/1k-users/training_movies.csv',
                    '../data/1k-users/training_ratings.csv',
                    '../data/1k-users/training_links.csv',
                )
    svd.configure(0.1, 0.15, 8)

    print 'Before function optimization:'
    print 'Training RMSE: %s' % svd.training_rmse
    print 'CV RMSE: %s' % svd.cross_validation_rmse

    svd.batch_gradient_descent()

    print 'After function optimization:'
    print 'Training RMSE: %s' % svd.training_rmse
    print 'CV RMSE: %s' % svd.cross_validation_rmse
