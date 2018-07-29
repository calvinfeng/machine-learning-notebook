from keras.utils import to_categorical
from sklearn import tree

import numpy as np
import csv


def main():
    x, y = [], []
    col_names = None
    with open('./datasets/credit_card_fraud.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        
        # Extract column names from header
        header = reader.next()
        col_names = header[1:len(header) - 1]
        
        for row in reader:
            x.append([float(el) for el in row[1:len(row)-1]])
            y.append(int(row[-1]))

    x = np.array(x).astype('float')
    y = np.array(y).astype('int')

    N = len(x)
    
    x_train = x[:int(N*0.80)]
    y_train = to_categorical(y[:int(N*0.80)], num_classes=2)
    
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)

    x_test = x[int(N*0.80):]
    y_test = to_categorical(y[int(N*0.80):], num_classes=2)

    print model.score(x_test, y_test)


if __name__ == '__main__':
    main()