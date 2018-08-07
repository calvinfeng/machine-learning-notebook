from models.loss import categorical_cross_entropy
from layers import Dense, ReLU, Softmax
from models import Sequential
from optimizers import GradientDescent

from keras.datasets import mnist
from keras.utils import to_categorical

from exceptions import Exception

import numpy as np


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print 'Imported MNIST data: training input %s and training labels %s.' % (x_train.shape, y_train.shape)
    print 'Imported MNIST data: test input %s and test labels %s.' % (x_test.shape, y_test.shape)

    N, H, W = x_train.shape
    x = x_train.reshape((N,H*W)).astype('float') / 255
    y = to_categorical(y_train, num_classes=10)

    model = Sequential()
    model.add(Dense(), ReLU(), layer_dim=(28*28, 300), weight_scale=1e-2)
    model.add(Dense(), ReLU(), layer_dim=(300, 100), weight_scale=1e-2)
    model.add(Dense(), Softmax(), layer_dim=(100, 10), weight_scale=1e-2)

    model.compile(optimizer=GradientDescent(learning_rate=1e-2),loss_func=categorical_cross_entropy)
    model.fit(x, y, epochs=10, batch_size=50, verbose=False)    

    N, H, W = x_test.shape
    x = x_test.reshape((N,H*W)).astype('float') / 255
    y = to_categorical(y_test, num_classes=10)

    model.evaluate(x, y)


if __name__ == '__main__':
    main()