import lowrank
import numpy as np
import csv


DATA_DIR = 'datasets/100k/'


def cosine_similarity(a, b):
    a_mag = (a * a).sum()
    b_mag = (b * b).sum()
    return a.dot(b) / (a_mag * b_mag)


def geometric_distance(a, b):
    diff = a - b
    return diff.dot(diff)


def movie_nearest_neighbor(target_movie_id):
    movies = dict()
    with open(DATA_DIR + 'movies.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # Skip the header
        for row in reader:
            movie_id, title, _ = row
            movies[movie_id] = title

    features = dict()
    with open(DATA_DIR + 'features.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # Skip the header
        for row in reader:
            movie_id = row[0]
            features[movie_id] = np.array(row[1:], dtype=float)

    distances = []
    for key in features:
        if key == target_movie_id:
            continue

        distances.append((key, geometric_distance(features[key], features[target_movie_id])))

    distances = sorted(distances, key=lambda x: x[1], reverse=False)
    print "Chosen movie is %s" % movies[target_movie_id]
    for item in distances[0:100]:
        print "Nearest neigbor: %s with distance %.3f" % (movies[item[0]], item[1])


def main():
    converter = lowrank.MatrixConverter(movies_filepath=DATA_DIR + 'movies.csv',
                                        ratings_filepath=DATA_DIR + 'ratings.csv')
    training_rating_mat, test_rating_mat = converter.get_rating_matrices()

    factorizer = lowrank.Factorizer(training_rating_mat, test_rating_mat, feature_dim=10, reg=0.05)
    benchmarks = factorizer.train(steps=400, learning_rate=1e-4)

    rmses = [bm[2] for bm in benchmarks]
    print rmses

    lowrank.unload_movie_features(filepath=DATA_DIR + 'features.csv',
                                  feature_dim=10,
                                  movie_features=converter.get_movie_feature_map(factorizer.M))
    # Let's pick Toy Story
    movie_nearest_neighbor('1')


if __name__ == '__main__':
    main()
