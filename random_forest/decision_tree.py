import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
from sklearn import tree
from sklearn import ensemble
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

N, H, W = x_train.shape
x = x_train.reshape((N,H*W)).astype('float') / 255
y = to_categorical(y_train, num_classes=10)

model = tree.DecisionTreeClassifier()
model.fit(x, y)

N, H, W = x_test.shape
x = x_test.reshape((N,H*W)).astype('float') / 255
y = to_categorical(y_test, num_classes=10)

print model.score(x, y)
