import tensorflow  as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

if __name__ == '__main__':
    ratings = tfds.load('movielens/100k-ratings', split='train')
    movies = tfds.load('movielens/100k-ratings', split='train')

    ratings = ratings.map(lambda x: { 'movie_title': x.movie_title() })
    print(ratings)
    print(movies)
    