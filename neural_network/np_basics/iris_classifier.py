"""Iris Classifier
This is a simple 1 input layer, 1 hidden layer and 1 output layer non-linear classifier for the famous iris data on
Kaggle

1st Gate: Multiplier (Dot product of input vector and first set of weights)
2nd Gate: Sigmoid (Produce activation vector that goes toward hidden layer)
3rd Gate: Multiplier (Dot product of first activation vector and second set of weights)
4th Gate: Sigmoid (Produce activation vector which is the prediction vector)
"""
import numpy as np
import csv
import matplotlib.pyplot as plt

from numpy.random import randn

"""Glossary
:param N: Number of training examples
:param input_dim: Number of parameters for each input vector
:param output_dim: Number of classifications, in this case we have 3
:param act_dim: Dimension of the activiation vector in our hidden layer
:param w1: The first weight matrix going from input layer to hidden layer
:param w2: The second weight matrix going from hidden layer to output layer
"""

# I am going to ignore bias units for the first implementation
N, input_dim, act_dim, output_dim = 150, 4, 5, 3
x, y = randn(N, input_dim), randn(N, output_dim) # Making a random training set
w1, w2 = randn(input_dim, act_dim), randn(act_dim, output_dim)

"""
Individual x input is (1 x 5) but the matrix is (200 x 5) i.e. row by col.
w1 is (5 x 5)

x * w1 => (200 x 5)(5 x 5) = (200 x 5)
sigmoid(x * w1) = a = (200 x 5)

a * w2 => (200 x 5)(5 x 3) = (200 x 3)
sigmoid(a * w2) = y_pred = (200 x 3)

We get 200 outputs in one go
"""

reader = csv.reader(open("datasets/iris.csv", "rt"), delimiter=",")
inputs, labels = [], []
for row in list(reader):
    inputs.append(row[:4])
    if row[4] == '0':
        labels.append([1, 0, 0])
    elif row[4] == '1':
        labels.append([0, 1, 0])
    elif row[4] == '2':
        labels.append([0, 0, 1])

iris_inputs, iris_labels = np.array(inputs).astype("float"), np.array(labels).astype("float")

x = iris_inputs
y = iris_labels
steps, losses = [], []
for step in range(40000):
    """Forward prop
    """
    theta_1 = x.dot(w1) # (N x input_dim)(input_dim x act_dim) = (N x act_dim)
    a = 1 / (1 + np.exp(-theta_1)) # (N x act_dim)
    theta_2 = a.dot(w2) # (N x act_dim)(act_dim x output_dim) => (N x output_dim)
    y_pred = 1 / (1 + np.exp(-theta_2)) # (N x output_dim)
    loss = np.square(y_pred - y).sum()
    print(step, loss)
    steps.append(step)
    losses.append(loss)

    """Backprop
    """
    grad_y_pred = 2.0 * (y_pred - y) # (N x output_dim) Gradient of y_pred w.r.t L
    grad_theta2 = grad_y_pred * (1 - y_pred) * y_pred # (N x output_dim) Gradient of theta2 w.r.t L
    grad_w2 = (grad_theta2.T.dot(a)).T # (output_dim x N)(N x act_dim) transpose => (act_dim x output_dim) Gradient of w2 w.r.t L

    grad_a = grad_theta2.dot(w2.T) # (N x output_dim)(output_dim x act_dim) => (N x act_dim)
    grad_theta1 = grad_a * (1 - a) * a # (N x act_dim)
    grad_w1 = (grad_theta1.T.dot(x)).T # (act_dim x N)(N x input_dim) transpose => (input_dim x act_dim)

    w2 -= 5e-3 * grad_w2
    w1 -= 5e-3 * grad_w1

plt.figure()
plt.plot(steps, losses)
plt.show()
