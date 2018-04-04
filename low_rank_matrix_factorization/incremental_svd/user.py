# Project: Recommender System
# Author(s): Calvin Feng

from random import random, sample
from pdb import set_trace as debugger
from math import sqrt

class User:
    def __init__(self, user_id, movie_ratings, preference_length, is_test_user=False):
        self.id = user_id
        self.preference_length = preference_length
        self.theta = self.random_init(preference_length)

        if is_test_user:
            self.set_ratings(movie_ratings, 2)
        else:
            self.set_ratings(movie_ratings, 0)

        self._baseline_rating = None

    def random_init(self, size):
        # Give User a bias term, which is 1
        preference_vector = [1]
        while len(preference_vector) < size:
            preference_vector.append(random())
        return preference_vector

    def set_ratings(self, movie_ratings, num_of_hidden_ratings):
        hidden_ratings = dict()
        if len(movie_ratings) >= num_of_hidden_ratings:

            random_keys = sample(movie_ratings, num_of_hidden_ratings)
            for i in range(0, num_of_hidden_ratings):
                key = random_keys[i]
                hidden_ratings[key] = movie_ratings.pop(key)

        self.movie_ratings = movie_ratings
        self.hidden_ratings = hidden_ratings

    @property
    def avg_rating(self):
        if self._baseline_rating is None and len(self.movie_ratings) != 0:
            avg = 0
            for movie_id in self.movie_ratings:
                avg += float(self.movie_ratings[movie_id])
            self._baseline_rating = avg / len(self.movie_ratings)

        return self._baseline_rating

    def sim(self, other_user):
        # Using Pearson correlation coefficient
        user_correlation = 0
        this_user_variance, other_user_variance = 0, 0
        movies_seen_by_both = []
        for movie_id in self.movie_ratings:
            if other_user.movie_ratings.get(movie_id):
                movies_seen_by_both.append(movie_id)

        if len(movies_seen_by_both) >= 20:
            for movie_id in movies_seen_by_both:
                this_rating, other_rating = float(self.movie_ratings[movie_id]), float(other_user.movie_ratings[movie_id])
                user_correlation += (this_rating - self.avg_rating)*(other_rating - other_user.avg_rating)
                this_user_variance += (this_rating - self.avg_rating)**2
                other_user_variance += (other_rating - other_user.avg_rating)**2
            if this_user_variance == 0 or other_user_variance == 0:
                # If one of the variances is zero, it's an undefined correlation
                return 0
            else:
                return user_correlation/(sqrt(this_user_variance)*sqrt(other_user_variance))
        else:
            # Statistically insignificant thus I return 0 for similarity
            return 0
