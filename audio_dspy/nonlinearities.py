import numpy as np


def soft_clipper(x, deg=3):
    """Implementation of a cubic soft clipper

    Parameters
    ----------
    x: {float, ndarray}
        input to the soft clipper
    deg: int, optional
        polynomial degree of the soft clipper

    Returns
    -------
    y: {float, ndarray}
        output of the soft clipper
    """
    assert deg % 2 == 1, "Degree must be odd integer"

    return np.where(x > 1, (deg-1)/deg,
                    np.where(x < -1, -(deg-1)/deg, x - x**deg/deg))


def hard_clipper(x):
    """Implementation of a hard clipper

    Parameters
    ----------
    x: {float, ndarray}
        input to the hard clipper

    Returns
    -------
    y: {float, ndarray}
        output of the hard clipper
    """
    return np.where(x > 1, 1,
                    np.where(x < -1, -1, x))


def dropout(x, width=0.5):
    """Implementation of dropout nonlinearity

    Parameters
    ----------
    x: {float, ndarray}
        input to the nonlinearity

    width: float, optional
        width of the dropout region

    Returns
    -------
    y: {float, ndarray}
        output of the nonlinearity
    """
    assert width > 0, "Width must be greater than zero"

    B = np.sqrt(width**3 / 3)
    return np.where(x > B, x - B + (B/width)**3,
                    np.where(x < -B, x + B - (B/width)**3,
                             (x/width)**3))


def halfWaveRect(x):
    """Implementation of an ideal half wave rectifier

    Parameters
    ----------
    x: {float, ndarray}
        input signal

    Returns
    -------
    y: {float, ndarray}
        output signal
    """
    return np.where(x < 0, 0, x)


def diodeRect(x, alpha=1.79, beta=0.2):
    """Implementation of a simple Schottky diode rectifier

    Parameters
    ----------
    x: {float, ndarray}
        input signal

    alpha: float
        input scale factor

    beta: float
        output scale factor

    Returns
    -------
    y: {float, ndarray}
        output signal
    """
    return beta * (np.exp(alpha*x) - 1.0)
