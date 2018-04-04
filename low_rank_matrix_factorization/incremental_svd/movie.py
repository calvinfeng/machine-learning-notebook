# Project: Recommender System
# Author(s): Calvin Feng

from numpy import dot
from random import random

class Movie:
    def __init__(self, movie_id, title, user_ratings, feature_length):
        self.id = movie_id
        self.title = title
        self.user_ratings = user_ratings
        self.feature_length = feature_length
        self.feature = self.random_init(feature_length)

    @property
    def viewer_ids(self):
        return self.user_ratings.keys()

    @property
    def avg_rating(self):
        if len(self.user_ratings) == 0:
            return 0

        rating_sum = 0
        for user_id in self.user_ratings:
            rating_sum += self.user_ratings[user_id]

        return rating_sum / len(self.user_ratings)

    def random_init(self, size):
        feature_vector = []
        while len(feature_vector) < size:
            feature_vector.append(random())
        return feature_vector

    def hypothesis(self, user):
        return dot(self.feature, user.theta)
