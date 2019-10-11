import numpy as np

def soft_clipper (x, deg=3):
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
    if isinstance (x, np.ndarray):
        t = np.copy (x)
        for n in range (len (x)):
            t[n] = soft_clipper (t[n], deg)
        return t
    
    if (x > 1): return (deg-1)/deg
    if (x < -1): return -(deg-1)/deg
    return x - x**deg/deg
