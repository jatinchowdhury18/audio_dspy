import numpy as np


def LMS(input, desired, mu, L):
    """
    Performs LMS adpative filtering on input signal

    Parameters:
    input : array-like
        Input signa;
    desired : array-like
        Desired signal
    mu : float
        Learning rate
    L : int
        Length of adaptive filter

    Return:
    y : array-like
        Filtered signal
    e : array-like
        Error signal
    w : array-like
        Final filter coefficients (of length L)
    """
    assert len(input) == len(
        desired), 'Desired and input signals must have equal length'

    N = len(input)

    w = np.zeros(L)
    y = np.zeros(N)
    x_win = np.zeros(L)
    e = np.zeros(N)

    for n in range(N):
        x_win = np.concatenate((x_win[1:L], [input[n]]))
        y[n] = np.dot(w, x_win)
        e[n] = desired[n] - y[n]
        w = w + mu * e[n] * x_win

    return y, e, w


def NLMS(input, desired, mu=0.1, L=7):
    """
    Performs Norm LMS adpative filtering on input signal

    Parameters:
    input : array-like
        Input signa;
    desired : array-like
        Desired signal
    mu : float
        Learning rate
    L : int
        Length of adaptive filter

    Return:
    y : array-like
        Filtered signal
    e : array-like
        Error signal
    w : array-like
        Final filter coefficients (of length L)
    """
    assert len(input) == len(
        desired), 'Desired and input signals must have equal length'
    N = len(input)

    w = np.zeros(L)
    y = np.zeros(N)
    x_win = np.zeros(L)
    e = np.zeros(N)

    for n in range(N):
        x_win = np.concatenate((x_win[1:L], [input[n]]))
        y[n] = np.dot(w, x_win)
        e[n] = desired[n] - y[n]
        w = w + mu * e[n] * x_win / np.sqrt(np.sum(x_win**2))

    return y, e, w


def NL_LMS(input, desired, mu, L, g, g_prime):
    """
    Performs Nonlinear LMS adaptive filtering on input signal

    Parameters:
    input : array-like
        Input signa;
    desired : array-like
        Desired signal
    mu : float
        Learning rate
    L : int
        Length of adaptive filter
    g : lambda (float) : float
        Nonlinear function, ex: tanh(x)
    g_prime : lambda (float) : float
        Derivative of nonlinear function, ex 1/cosh(x)^2

    Return:
    y : array-like
        Filtered signal
    e : array-like
        Error signal
    w : array-like
        Final filter coefficients (of length L)
    """
    assert len(input) == len(
        desired), 'Desired and input signals must have equal length'

    N = len(input)

    w = np.zeros(L)
    y = np.zeros(N)
    x_win = np.zeros(L)
    e = np.zeros(N)

    for n in range(N):
        x_win = np.concatenate((x_win[1:L], [input[n]]))
        y[n] = np.dot(w, x_win)
        e[n] = desired[n] - g(y[n])
        w = w + mu * e[n] * x_win * g_prime(y[n])

    return y, e, w
