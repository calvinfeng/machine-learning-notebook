from csv_util import load_movies, load_user_ratings
import numpy as np


class MatrixConverter(object):
    TEST_TO_TRAINING_RATIO = 0.10

    def __init__(self, movies_filepath, ratings_filepath):
        np.random.seed(0)
        self.movie_dict = load_movies(movies_filepath)
        self.training_rating_dict = load_user_ratings(ratings_filepath)
        self.test_rating_dict = dict()

        training_count, test_count = 0, 0

        i, self.user_id_to_idx, self.user_idx_to_id = 0, dict(), dict()
        for user_id in self.training_rating_dict:
            self.user_id_to_idx[user_id] = i
            self.user_idx_to_id[i] = user_id
            i += 1

            if self.test_rating_dict.get(user_id) is None:
                self.test_rating_dict[user_id] = dict()

            for movie_id in self.training_rating_dict[user_id]:
                if np.random.uniform() <= self.TEST_TO_TRAINING_RATIO:
                    self.test_rating_dict[user_id][movie_id] = self.training_rating_dict[user_id][movie_id]
                    test_count += 1
                else:
                    training_count += 1

        j, self.movie_id_to_idx, self.movie_idx_to_id = 0, dict(), dict()
        for movie_id in self.movie_dict:
            self.movie_id_to_idx[movie_id] = j
            self.movie_idx_to_id[j] = movie_id
            j += 1

        print "CSV data are loaded with %d training samples and %d test samples from %d users on %d movies" % (
            training_count, test_count, len(self.user_id_to_idx), len(self.movie_id_to_idx))

    def get_rating_matrices(self):
        """Returns two heavily sparsed numpy 2D arrays containing ratings from each user i to movie j. If a rating is
        missing, it is filled with zero. The first one is for training and the last one is for testing.
        """
        training_mat, test_mat = [], []
        for i in range(len(self.user_idx_to_id)):
            training_row, test_row = [], []
            user_id = self.user_idx_to_id[i]
            for j in range(len(self.movie_idx_to_id)):
                movie_id = self.movie_idx_to_id[j]

                # If user hasn't rated a movie, append 0 instead of None
                if self.training_rating_dict[user_id].get(movie_id) is None:
                    training_row.append(0)
                    test_row.append(0)
                    continue

                # If a rating is labeled as test data, do not append it to the training matrix
                if self.test_rating_dict[user_id].get(movie_id) is None:
                    training_row.append(self.training_rating_dict[user_id][movie_id])
                    test_row.append(0)
                else:
                    training_row.append(0)
                    test_row.append(self.test_rating_dict[user_id][movie_id])

            training_mat.append(training_row)
            test_mat.append(test_row)

        return np.array(training_mat), np.array(test_mat)

    def get_movie_feature_map(self, movie_latent_mat):
        """Returns a dictionary of movie ID mapping to the movie's latent feature vector

        :param numpy.array movie_latent_mat: Factorized low rank matrix of movies to latent features
        """
        feature_map = dict()
        M, _ = movie_latent_mat.shape
        for m in range(M):
            movieId = self.movie_idx_to_id[m]
            feature_map[movieId] = movie_latent_mat[m]

        return feature_map

    def export_movie_latent_matrix(self, movie_latent_mat, filepath):
        """Exports movie latent matrix to a CSV file where each row starts with movie ID and the rest are feature values
        seperated by comma
        :param numpy.array movie_latent_mat: Factorized low rank matrix of movies to latent features
        :param string filepath: Path for exporting CSV
        """
        feature_map = dict()
        M, K = movie_latent_mat.shape
        for m in range(M):
            movieId = self.movie_idx_to_id[m]
            feature_map[movieId] = movie_latent_mat[m]

        unload_movie_features(filepath, feature_dim=K, movie_features=feature_map)
