import numpy as np
import neural_net.data_util as dutil
from neural_net.network import NeuralNetwork


if __name__ == "__main__":
    data_dir = 'datasets/cifar-10-batches-py'

    xtr, ytr, xval, yval, xte, yte = dutil.preprocess_cifar_10(data_dir)
    print 'Train data shape: %s' % str(xtr.shape)
    print 'Train labels shape: %s' % str(ytr.shape)
    print 'Validation data shape: %s' % str(xval.shape)
    print 'Validation labels shape: %s' % str(yval.shape)
    print 'Test data shape: %s' % str(xte.shape)
    print 'Test labels shape: %s' % str(yte.shape)

    N = xtr.shape[0] # Number of training examples
    input_dim = xtr.shape[1] # Number of pixels as input
    hidden_dim = 200
    output_dim = 10 # Number of classes

    network = NeuralNetwork(input_dim, hidden_dim, output_dim, std=1e-2)

    train_acc = (network.predict(xtr) == ytr).mean()
    print 'Training accuracy: %s' % str(train_acc)

    loss_hist = network.train(xtr, ytr, learning_rate=1e-2, reg=0.25)

    train_acc = (network.predict(xtr) == ytr).mean()
    print 'Training accuracy: %s' % str(train_acc)

    val_acc = (network.predict(xval) == yval).mean()
    print 'Validation accuracy: %s' % str(val_acc)

    test_acc = (network.predict(xte) == yte).mean()
    print 'Test accuracy: %s' % str(test_acc)
