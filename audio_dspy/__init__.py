import numpy as np
from .transfer_function_tools import *
from .prony import *
from .sweeps import *
from .nonlinearities import *
from .plotting import *
from .eq_design import *
name = "audio_dspy"


def normalize(x):
    """Normalize an array of data (real or complex)

    Parameters
    ----------
    x : array-like
        Data to be normalized

    Returns
    -------
    y : array-like
        Normalized data
    """
    return x / np.max(np.abs(x))


def impulse(N):
    """Create an impulse of length N

    Parameters
    ----------
    N : int
        Length of the impulse

    Returns
    -------
    h : array-like
        Generated impulse response
    """
    h = np.zeros(N)
    h[0] = 1.0
    return h
