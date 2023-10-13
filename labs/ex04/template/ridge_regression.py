# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    # ridge regression: TODO
    tx_transpose = np.transpose(tx)
    lambda_ = 2 * len(y) * lambda_
    left = tx_transpose@tx + np.eye(tx.shape[1]) * lambda_
    right = tx_transpose@y

    """"# Version WITHOUT inversion
    return np.linalg.lstsq(left, right, rcond=None)[0]"""

    # Version WITH inversion
    return np.linalg.inv(left) @ right
    # ***************************************************
