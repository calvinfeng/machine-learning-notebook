import numpy as np


def loss(U, M, R, reg=0.0):
    """Compute loss

    :param U: User latent feature matrix, there are I movies and K features
    :type U: numpy 2-d array
    :param M: Movie latent feature matrix, there are J movies and K features
    :type M: numpy 2-d array
    :param R: Rating matrix, i-index represents user and j-index represents movie
    :type R: numpy 2-d array
    :param reg: Regularization strength
    :type reg: float
    """
    diff = np.dot(U, M.T) - R
    loss = 0.5 * np.sum(diff * diff)
    loss += reg * np.sum(U * U) / 2
    loss += reg * np.sum(M * M) / 2
    return loss


def rel_error(X, Y):
    """Compute maximum relative error

    :param X: Matrix of the same shape as Y
    :type X: numpy array
    :param Y: Matrix of the same shape as X
    :type Y: numpy array
    """
    return np.max(np.abs(X - Y) / (np.maximum(1e-8, np.abs(X) + np.abs(Y))))


def compute_grad(U, M, R, reg=0.0):
    """Compute gradients for U and M

    :param U: User latent feature matrix, there are I movies and K features
    :type U: numpy 2-d array
    :param M: Movie latent feature matrix, there are J movies and K features
    :type M: numpy 2-d array
    :param R: Rating matrix, i-index represents user and j-index represents movie
    :type R: numpy 2-d array
    :param reg: Regularization strength
    :type reg: float
    """
    u_grad = np.zeros(U.shape)
    m_grad = np.zeros(M.shape)

    num_user, lat_dim = U.shape
    num_movie, lat_dim = M.shape

    diff = np.dot(U, M.T) - R
    for i in range(num_user):
        u_grad[i] = np.sum(diff[i].reshape(num_movie, 1) * M, axis=0) + (reg * U[i])

    for j in range(num_movie):
        m_grad[j] = np.sum(diff.T[j].reshape(num_user, 1) * U, axis=0) + (reg * M[j])

    return u_grad, m_grad


def compute_grad_vectorized(U, M, R, reg=0.0):
    """Compute gradients for U and M

    :param U: User latent feature matrix, there are I movies and K features
    :type U: numpy 2-d array
    :param M: Movie latent feature matrix, there are J movies and K features
    :type M: numpy 2-d array
    :param R: Rating matrix, i-index represents user and j-index represents movie
    :type R: numpy 2-d array
    :param reg: Regularization strength
    :type reg: float
    """
    grad_out = np.dot(U, M.T) - R
    grad_u = np.dot(grad_out, M)+ (reg * U)
    grad_m = np.dot(grad_out.T, U) + (reg * M)
    return grad_u, grad_m


def compute_num_grad(U, M, R, loss_func, reg=0.0, h=1e-5):
    """Compute numerical gradients for U and M

    :param U: User latent feature matrix, there are I movies and K features
    :type U: numpy 2-d array
    :param M: Movie latent feature matrix, there are J movies and K features
    :type M: numpy 2-d array
    :param R: Rating matrix, i-index represents user and j-index represents movie
    :type R: numpy 2-d array
    :param reg: Regularization strength
    :type reg: float
    """
    num_grad_u = np.zeros(U.shape)
    num_grad_m = np.zeros(M.shape)

    U_dim, L_dim = U.shape
    M_dim, L_dim = M.shape

    for i in range(U_dim):
        for k in range(L_dim):
            old_val = U[i][k]

            U[i][k] = old_val + h
            fuph = loss_func(U, M, R, reg)

            U[i][k] = old_val - h
            fumh = loss_func(U, M, R, reg)

            U[i][k] = old_val
            num_grad_u[i][k] = (fuph - fumh) / (2 * h)

    for j in range(M_dim):
        for k in range(L_dim):
            old_val = M[j][k]

            M[j][k] = old_val + h
            fmph = loss_func(U, M, R, reg)

            M[j][k] = old_val - h
            fmmh = loss_func(U, M, R, reg)

            M[j][k] = old_val
            num_grad_m[j][k] = (fmph - fmmh) / (2 * h)

    return num_grad_u, num_grad_m
