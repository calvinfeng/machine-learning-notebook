import cPickle as pickle
import numpy as np
import platform
import os


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """Load a single batch of CIFAR-10 data
    """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(dir):
    """Load all CIFAR-10 data which contains 5 batches each has 10,000 images
    """
    xs = []
    ys = []
    for b in range(1, 6):
        filepath = os.path.join(dir, 'data_batch_%d' % b)
        X, Y = load_CIFAR_batch(filepath)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def preprocess_cifar_10(dir, num_training=49000, num_validation=1000, num_test=1000):
    """Load the CIFAR-10 dataset from disk and perform preprocessing
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation)) # Between 49000 and 50000
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_training)) # Between 0 to 49000
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = list(range(num_test)) # Anything from 0 to 10000
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Notice that our neural network does not take rank 3 tensor as input, we must
    # reshape the 32x32x3 into a single row that is 32x32x3 long which is 3072
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    x_training, y_training, x_test, y_test = load_CIFAR10('data/cifar-10-batches-py')
    print x_training.shape
    print y_training.shape
