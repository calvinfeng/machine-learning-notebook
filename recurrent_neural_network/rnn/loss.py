import numpy as np
from gradient_check import eval_numerical_gradient


def rel_error(x, y):
    """Returns relative error"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    

def temporal_softmax_loss(score, y, mask, verbose=False):
    """
    Args:
        score (np.array): Input scores, of shape (N, T, V)
        y (np.array): Ground truth indices, of shape (N, T) where each element is in range 
                      0 <= y[n, t] < V
        mask (np.array): Boolean array of shape (N, T) where mask[n, t] tells whehter or not the 
                         score at x[n, t] should contribute to the loss.

    Returns tuple:
        - loss (float): Scalar giving loss
        - grad_score (np.array): Gradient of loss with respect to scores
    """
    N, T, V = score.shape
    
    # Reshape the inputs to flatten them
    score_flat = score.reshape(N*T, V)
    y_flat = y.reshape(N*T)
    mask_flat = mask.reshape(N*T)

    # Compute probabilities
    probs = np.exp(score_flat - np.max(score_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)

    # Compute loss
    ################################################################################################
    # Mask is composed of boolean values, i.e. True/False. When you multiply a numpy array with a 
    # mask (which is just another numpy array), the false(s) will act as zeros.
    # Example:
    # mask = np.array([True, True, False, False])
    # nums = np.array([1, 1, 1, 1])
    # mask * nums => array([1, 1, 0, 0])
    ################################################################################################
    loss = -1 * np.sum(mask_flat * np.log(probs[np.arange(N*T), y_flat])) / N

    grad_score_flat = probs.copy()
    grad_score_flat[np.arange(N*T), y_flat] -= 1
    grad_score_flat /= N
    grad_score_flat *= mask_flat[:, None] # Needed to do this for broadcasting

    if verbose:
        print 'gradient of loss w.r.t score flat dimension:', grad_score_flat.shape
    
    grad_score = grad_score_flat.reshape(N, T, V)

    return loss, grad_score



def check_loss(N, T, V, p):
    x = 0.001 * np.random.randn(N, T, V)
    y = np.random.randint(V, size=(N, T))
    mask = np.random.rand(N, T) <= p
    print(temporal_softmax_loss(x, y, mask)[0])


def main():
    N, T, V = 100, 1, 10
    check_loss(100, 1, 10, 1.0)   # Should be about 2.3
    check_loss(100, 10, 10, 1.0)  # Should be about 23
    check_loss(5000, 10, 10, 0.1) # Should be about 2.3

    # Gradient check for temporal softmax loss
    N, T, V = 7, 8, 9

    x = np.random.randn(N, T, V)
    y = np.random.randint(V, size=(N, T))
    mask = (np.random.rand(N, T) > 0.5)

    loss, grad_x = temporal_softmax_loss(x, y, mask, verbose=False)

    grad_x_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], x, verbose=False)
    
    print rel_error(grad_x, grad_x_num)
    print loss


if __name__ == "__main__":
    main()