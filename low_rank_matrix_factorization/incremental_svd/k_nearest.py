# Project: Recommender System
# Author(s): Calvin Feng

from movie import Movie
from user import User
from data_reducer import DataReducer
from progress import Progress
from math import sqrt
from pdb import set_trace
from random import uniform

class KNearest:

    def __init__(self, movie_csv_filepath, rating_csv_filepath, link_csv_filepath):
        reducer = DataReducer(movie_csv_filepath, rating_csv_filepath, link_csv_filepath)

        latent_factor_length = 0

        self.movies = dict()
        for movie_id in reducer.movies:
            movie = reducer.movies[movie_id]
            self.movies[movie_id] = Movie(movie_id, movie['title'], movie['user_ratings'], latent_factor_length)

        self.users = dict()
        for user_id in reducer.users:
            user = reducer.users[user_id]
            self.users[user_id] = User(user_id, user['movie_ratings'], latent_factor_length, True)

    def hypothesis(self, user, movie):
        neighbor_ids = movie.user_ratings.keys()
        score = 0
        sim_norm = 0
        for id in neighbor_ids:
            neighbor = self.users[id]
            sim = user.sim(neighbor)
            if (sim > 0.70 or sim < -0.70)  and neighbor.movie_ratings.get(movie.id):
                score += sim * (float(neighbor.movie_ratings[movie.id]) - float(neighbor.avg_rating))
                sim_norm += abs(sim)

        if sim_norm == 0:
            return 'Insufficient information'

        return user.avg_rating + (score / sim_norm)

    @property
    def rmse(self):
        sq_error = 0
        m = 0
        for user_id in self.users:
            user = self.users[user_id]
            for movie_id in user.hidden_ratings:
                movie = self.movies[movie_id]
                predicted_rating = self.hypothesis(user, movie)
                if isinstance(predicted_rating, float):
                    sq_error += (self.hypothesis(user, movie) - float(user.hidden_ratings[movie_id]))**2
                    m += 1
                #
                # sq_error += (uniform(0.5, 5) - float(user.hidden_ratings[movie_id]))**2
                # m += 1


        return sqrt(sq_error / m)

if __name__ == '__main__':
    knn = KNearest(
            '../data/20k-users/training_movies.csv',
            '../data/20k-users/training_ratings.csv',
            '../data/20k-users/training_links.csv',
        )

    print knn.rmse
