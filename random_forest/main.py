from keras.utils import to_categorical
from sklearn import tree
from forest import DecisionTree, RandomForest

import numpy as np
import csv


IRIS_MAP = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}


def predict_with_sklearn_dtree(data):
    N = len(data)

    x, y = None, None
    for row in data:
        num_feat = len(row) - 1
        if x is None and y is None:
            x = np.array(row[:num_feat])
            y = np.array([IRIS_MAP[row[-1]]])
    
        x = np.vstack((x, np.array(row[:num_feat])))
        y = np.append(y, IRIS_MAP[row[-1]])

    x_train = x[:int(0.80*N)]
    y_train = to_categorical(y[:int(0.80*N)], num_classes=3)
    
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)

    x_test = x[int(0.80*N):]
    y_test = to_categorical(y[int(0.80*N):], num_classes=3)

    return model.score(x_test, y_test)


def predict_with_dtree(col_names, data):
    N = len(data)
    training_data = data[:int(0.80*N)]

    tree = DecisionTree()
    tree.fit(col_names, training_data)

    testing_data = data[int(0.80*N):]

    for row in testing_data:
        print 'Actual label is %s and predicted %s' % (row[-1], tree.predict(row))


def predict_with_forest(col_names, data):
    N = len(data)
    training_data = data[:int(0.80*N)]

    forest = RandomForest(n_samples=10, n_features=3)
    forest.fit(col_names, training_data)

    testing_data = data[int(0.80*N):]

    for row in testing_data:
        print 'Actual label is %s and predicted %s' % (row[-1], forest.predict(row))


def load_data(csv_path):
    col_names = None
    data = []

    with open(csv_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)

        # Extract column names from header
        header = reader.next()
        col_names = header[1:len(header)]

        for row in reader:
            data_row = [float(el) for el in row[1:len(row)-1]] + [row[-1]]
            data.append(data_row)

    return col_names, data


def main():
    col_names, data = load_data('./datasets/iris.csv')
    
    # Shuffle the data
    np.random.shuffle(data)

    # print predict_with_sklearn_dtree(data)
    # predict_with_dtree(col_names, data)
    predict_with_forest(col_names, data)


if __name__ == '__main__':
    main()