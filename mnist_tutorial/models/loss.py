import numpy as np


def categorical_cross_entropy(y_pred, y):
    """Computes categorical cross entropy loss.
    Args:
        y_pred (numpy.ndarray): Output of the network, of shape (N, C) where x[i, j] is the softmax 
                                probability for for jth class for the ith input.
        y (numpy.ndarray): Vector of labels in one-hot representation.
    Returns:
        loss (float): Scalar value of the cross entropy loss.
    """
    N = len(y_pred)
    y = np.argmax(y, axis=1)
    log_probs = np.log(y_pred)

    return -1 * np.sum(log_probs[np.arange(N), y]) / N


def softmax(x, y):
    """Computes softmax cross entropy loss and gradient
    Args:
        x (numpy.ndarray): Input data, of shape (N, C) where x[i, j] is the score for jth class for
                           the ith input.
        y (numpy.ndarray): Vector of labels in one-hot representation.
    Returns:
        loss (float): Scalar value of the loss
        grad_x (numpy.ndarray): Gradient of the loss with respect to x
    """
    # Ensure numerical stability by shifting the input matrix by its largest value in each row.
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    # Compute loss    
    N = x.shape[0]
    y = np.argmax(y, axis=1)
    loss = -np.sum(log_probs[np.arange(N), y]) / N

    # Compute gradients
    grad_x = probs.copy()
    grad_x[np.arange(N), y] -= 1
    grad_x /= N

    return loss, grad_x, probs