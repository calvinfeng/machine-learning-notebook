import csv
import numpy as np


def load_iris_data(filepath, multi_dimen_labels=True):
    with open(filepath, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        inputs, labels = [], []
        for row in reader:
            inputs.append(row[:4])
            if multi_dimen_labels:         
                if row[4] == '0':
                    labels.append([1, 0, 0])
                elif row[4] == '1':
                    labels.append([0, 1, 0])
                elif row[4] == '2':
                    labels.append([0, 0, 1])
            else:
                labels.append(float(row[4]))

    if multi_dimen_labels:
        return np.array(inputs).astype('float'), np.array(labels).astype('float')
    
    return np.array(inputs).astype('float'), np.array(labels).astype('int')

