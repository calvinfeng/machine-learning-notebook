import numpy as np
from six.moves import cPickle as pickle
import platform
import os


def get_preprocessed_CIFAR10(cifar_10_dir,
                            num_training=49000, num_validation=1000, num_test=1000,
                            should_subtract_mean=True,
                            should_transpose=True):
    """Load CIFAR-10 dataset from disk and perform pre-processing to prepare it for model training
    """
    X_train, y_train, X_test, y_test = _load_CIFAR10(cifar_10_dir)

    # Subsample for validation set
    mask = list(range(num_training, num_training + num_validation))
    X_val, y_val = X_train[mask], y_train[mask]

    mask = list(range(num_training))
    X_train, y_train = X_train[mask], y_train[mask]

    mask = list(range(num_test))
    X_test, y_test = X_test[mask], y_test[mask]

    if should_subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    if should_transpose:
        X_train = X_train.transpose(0, 3, 1, 2).copy()
        X_val = X_val.transpose(0, 3, 1, 2).copy()
        X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

def _load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("Invalid Python version: {}".format(version))


def _load_CIFAR_batch(filename):
    """Load a single batch of CIFAR data
    """
    with open(filename, 'rb') as f:
        datadict = _load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y


def _load_CIFAR10(ROOT):
    """Load all CIFAR dat from a root directory
    """
    Xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = _load_CIFAR_batch(f)
        Xs.append(X)
        ys.append(Y)

    X_train = np.concatenate(Xs)
    Y_train = np.concatenate(ys)

    del X, Y
    X_test, Y_test = _load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return X_train, Y_train, X_test, Y_test
