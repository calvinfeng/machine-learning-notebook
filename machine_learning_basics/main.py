import numpy as np

from layers.dense import Dense
from layers.activations import ReLU


def main():
    weight_std = 0.01
    params = {
        "dense_1/kernel": weight_std * np.random.randn(1, 6),
        "dense_1/bias": np.zeros((6,)),
        "dense_2/kernel": weight_std * np.random.randn(6, 3),
        "dense_2/bias": np.zeros((3,)),
        "dense_3/kernel": weight_std * np.random.randn(3, 1),
        "dense_3/bias": np.zeros((1,)),
    }
    dense_1 = Dense()
    relu_1 = ReLU()
    dense_2 = Dense()
    relu_2 = ReLU()
    dense_3 = Dense()
    relu_3 = ReLU()


if __name__ == '__main__':
    main()