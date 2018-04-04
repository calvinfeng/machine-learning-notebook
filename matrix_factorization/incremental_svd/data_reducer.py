# Project: Recommender System
# Author(s): Calvin Feng

from csv import reader
from csv import writer
from sets import Set
from pdb import set_trace as debugger

class DataReducer:

    def __init__(self, movies_file_path, ratings_file_path, links_file_path):
        # File paths
        self.movies_file_path = movies_file_path
        self.ratings_file_path = ratings_file_path
        self.links_file_path = links_file_path

        # Properties
        self._movies = None
        self._users = None

        self._user_training_set = None
        self._movie_training_set = None

    @property
    def movies(self) :
        if self._movies is None:
            # First, load movies
            movie_dict = dict()
            csv = reader(open(self.movies_file_path))
            for row in csv:
                if row[0].isdigit():
                    movie_id, movie_title, movie_year = row[0], row[1], row[2]
                    movie_dict[movie_id] = {'title': movie_title, 'year': movie_year}

            rating_count = 0

            # Then, load historical user ratings
            user_dict = dict()
            csv = reader(open(self.ratings_file_path))
            for row in csv:
                if row[0].isdigit():
                    rating_count += 1
                    user_id, movie_id, rating = row[0], row[1], row[2]
                    if movie_dict[movie_id].get('user_ratings'):
                        movie_dict[movie_id]['user_ratings'][user_id] = rating
                    else:
                        movie_dict[movie_id]['user_ratings'] = { user_id: rating }

                    if user_dict.get(user_id):
                        user_dict[user_id]['movie_ratings'][movie_id] = rating
                    else:
                        user_dict[user_id] = dict()
                        user_dict[user_id]['movie_ratings'] = { movie_id: rating }

            # Store them
            self._users = user_dict
            self._movies = movie_dict
            self.rating_count = rating_count

        return self._movies

    @property
    def users(self):
        if self._users is None:
            rating_count = 0
            user_dict = dict()
            csv = reader(open(self.ratings_file_path))
            for row in csv:
                if row[0].isdigit():
                    rating_count += 1
                    user_id, movie_id, rating = row[0], row[1], row[2]
                    if user_dict.get(user_id):
                        user_dict[user_id]['movie_ratings'][movie_id] = rating
                    else:
                        user_dict[user_id] = dict()
                        user_dict[user_id]['movie_ratings'] = { movie_id: rating }
            self._users = user_dict
            self.rating_count = rating_count

        return self._users

    '''
    API for saving training_set or test_set
    '''
    def export_training_set(self, starting_user_id, max_user_count, dir):
        is_rating_export_successful = self._export_movie_ratings(starting_user_id, max_user_count, dir)
        is_movie_export_successful = self._export_movies(dir)
        is_link_export_successful = self._export_movie_links(dir)
        if (is_rating_export_successful and is_movie_export_successful) and is_link_export_successful:
            print 'Success!'

    '''
    Private methods
    '''
    # Set constraint on number of users
    def _export_movie_ratings(self, starting_user_id, max_user_count, dir):
        csv = reader(open(self.ratings_file_path))
        movie_training_set, user_training_set = Set(), Set()
        with open(dir + '/training_ratings.csv', 'wt') as outfile:
            output = writer(outfile)
            output.writerow(('userId', 'movieId', 'rating', 'timestamp'))
            write_count = 0
            for row in csv:
                if len(user_training_set) == max_user_count and row[0] not in user_training_set:
                    self._movie_training_set = movie_training_set
                    self._user_training_set = user_training_set
                    print 'Exported rating count: %s from %s users' % (write_count, len(self._user_training_set))
                    return True

                if row[0].isdigit():
                    user_id = row[0]
                    if int(user_id) >= starting_user_id:
                        output.writerow((row[0], row[1], row[2], row[3]))
                        movie_training_set.add(row[1])
                        user_training_set.add(row[0])
                        write_count += 1



        self._movie_training_set = movie_training_set
        self._user_training_set = user_training_set
        return False

    def _export_movies(self, dir):
        if self._movie_training_set is None:
            return False

        csv = reader(open(self.movies_file_path))
        with open(dir + '/training_movies.csv', 'wt') as outfile:
            output = writer(outfile)
            output.writerow(('movieId', 'title', 'year', 'popularity', 'genres'))
            write_count = 0
            for row in csv:
                if row[0].isdigit():
                    movie_id = row[0]
                    if movie_id in self._movie_training_set:
                        title = row[1].strip()
                        year = title[len(title) - 5: len(title) - 1]
                        genres = row[2]
                        title = title[:len(title) - 6].strip()
                        popularity = len(self.movies[movie_id]['user_ratings'])
                        output.writerow((movie_id, title, year, popularity, genres))
                        write_count += 1
            if write_count == len(self._movie_training_set):
                print 'Exported movie count: %s' % write_count
                return True
            return False

    def _export_movie_links(self, dir):
        if self._movie_training_set is None:
            return False

        csv = reader(open(self.links_file_path))
        with open(dir + '/training_links.csv', 'wt') as outfile:
            output = writer(outfile)
            output.writerow(('movieId', 'imdbId', 'tmdbId'))
            write_count = 0
            for row in csv:
                if row[0].isdigit() and row[0] in self._movie_training_set:
                    output.writerow((row[0], "tt" + row[1], row[2]))
                    write_count += 1
            if write_count == len(self._movie_training_set):
                return True
            return False


if __name__ == '__main__':
    reducer = DataReducer('../data/full/movies.csv', '../data/full/ratings.csv', '../data/full/links.csv')

    # for movie_id in reducer.movies:
    #     movie = reducer.movies[movie_id]
    #     if movie.get('user_ratings') is None:
    #         print '%s has no ratings' % movie['title']
    #     elif len(movie['user_ratings']) > 5000:
    #         print '%s has %s ratings' % (movie['title'], len(movie['user_ratings']))
    #
    # for user_id in reducer.users:
    #     user = reducer.users[user_id]
    #     if len(user) > 300:
    #         print 'User %s has %s ratings' % (user_id, len(user['movie_ratings']))
    #
    # rating_count = reducer.rating_count
    # print 'Total rating count: %s' % (rating_count)

    # Starting user_id is 200,000 and we are exporting 20,000 users
    print reducer.export_training_set(200000, 20000, '../data/20k-users')

    # Test set is similar to training set
    # print reducer.export_training_set(230000, 1000, '../data/1k-users')

    # print reducer.export_training_set(190000, 10, '../data/10-users')
