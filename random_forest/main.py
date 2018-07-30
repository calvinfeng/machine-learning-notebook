from keras.utils import to_categorical
from sklearn import tree
from random_forest import build_tree, print_tree

import numpy as np
import csv


def predict_with_sklearn_dtree(x_rows, y_labels):
    x = np.array(x_rows).astype('float')
    y = np.array(y_labels).astype('int')

    N = len(x)
    
    x_train = x[:int(N*0.80)]
    y_train = to_categorical(y[:int(N*0.80)], num_classes=2)
    
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)

    x_test = x[int(N*0.80):]
    y_test = to_categorical(y[int(N*0.80):], num_classes=2)

    return model.score(x_test, y_test)


def predict_with_dtree(col_names, x_rows, y_labels):
    training_data = []

    N = len(x_rows)
    for i in range(int(N*0.80)):
        training_data.append(x_rows[i] + [y_labels[i]])
    
    root = build_tree(col_names, training_data)
    print_tree(root)

    testing_data = []
    for i in range(int(N*0.80), N):
        testing_data.append(x_rows[i], [y_labels[i]])


def main():
    col_names = None
    x, y = [], []

    with open('./datasets/credit_card_fraud.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        
        # Extract column names from header
        header = reader.next()
        col_names = header[1:len(header) - 1]
        
        for row in reader:
            x.append([float(el) for el in row[1:len(row)-1]])
            y.append(int(row[-1]))
    
    predict_with_dtree(col_names, x, y)


if __name__ == '__main__':
    main()