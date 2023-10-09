# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    N = x.shape[0]
    poly = np.ones(N).reshape((N, 1))

    for deg in range(degree):
        deg += 1
        poly = np.hstack((poly, (x ** deg).reshape(N, 1)))

    return poly
    # ***************************************************
