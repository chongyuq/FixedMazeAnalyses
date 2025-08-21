from scipy.special import softmax
import numpy as np


def accuracy_with_fixed_constant(X, y, c, w):
    """

    :param X: The X matrix is number of observations x actions x features. The actions that
    #  are not possible are changed to - inf so not to be taken into account
    :type X: numpy array
    :param y: actual observations
    :type y: numpy array
    :param w: shape no. of features - this is to be optimized
    :type w: numpy array
    :return: log likelihood of observations given w
    :rtype:
    """
    y_pred = softmax(X @ w + c, axis=-1)
    accuracy = (y_pred.argmax(axis=-1) == y).sum() / len(y)
    return accuracy


def log_likelihood(X, y, w):
    """

    :param X: The X matrix is number of observations x actions x features. The actions that
    #  are not possible are changed to - inf so not to be taken into account
    :type X: numpy array
    :param y: actual observations
    :type y: numpy array
    :param w: shape no. of features - this is to be optimized
    :type w: numpy array
    :return: log likelihood of observations given w
    :rtype:
    """
    y_pred = softmax(X @ w, axis=-1)
    loglikelihood = - np.log(y_pred[range(len(y)), y]+1e-30).sum()
    return loglikelihood


def log_likelihood_with_fixed_constant(X, y, c, w):
    """

    :param X: The X matrix is number of observations x actions x features. The actions that
    #  are not possible are changed to - inf so not to be taken into account
    :type X: numpy array
    :param y: actual observations
    :type y: numpy array
    :param w: shape no. of features - this is to be optimized
    :type w: numpy array
    :return: log likelihood of observations given w
    :rtype:
    """
    y_pred = softmax(X @ w + c, axis=-1)
    loglikelihood = - np.log(y_pred[range(len(y)), y]+1e-30).mean()
    return loglikelihood