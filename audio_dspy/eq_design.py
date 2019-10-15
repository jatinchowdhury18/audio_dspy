import numpy as np

def design_bell (fc, Q, gain, fs):
    """Calculates filter coefficients for a bell filter.

    Parameters
    ----------
    fc : float
        Center frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    gain : float
        Linear gain for the center frequency of the filter
    fs :  float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    wc = 2 * np.pi * fc / fs
    c = 1.0 / np.tan (wc / 2.0)
    phi = c*c
    Knum = c / Q
    Kdenom = Knum

    if (gain > 1.0):
        Knum *= gain
    elif (gain < 1.0):
        Kdenom /= gain

    a0 = phi + Kdenom + 1.0

    b = [(phi + Knum + 1.0) / a0, 2.0 * (1.0 - phi) / a0, (phi - Knum + 1.0) / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - Kdenom + 1.0) / a0]

    return np.asarray (b), np.asarray(a)

def design_LPF2 (fc, Q,  fs):
    """Calculates filter coefficients for a 2nd-order lowpass filter

    Parameters
    ----------
    fc : float
        Cutoff frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    fs : float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    wc = 2 * np.pi * fc / fs
    c = 1.0 / np.tan (wc / 2.0)
    phi = c*c
    K = c / Q
    a0 = phi + K + 1.0

    b = [1 / a0, 2.0 / a0, 1.0 / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - K + 1.0) / a0]

    return np.asarray (b), np.asarray(a)

def design_HPF2 (fc, Q,  fs):
    """Calculates filter coefficients for a 2nd-order highpass filter

    Parameters
    ----------
    fc : float
        Cutoff frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    fs : float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    wc = 2 * np.pi * fc / fs
    c = 1.0 / np.tan (wc / 2.0)
    phi = c*c
    K = c / Q
    a0 = phi + K + 1.0

    b = [phi / a0, -2.0 * phi / a0, phi / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - K + 1.0) / a0]
    return np.asarray (b), np.asarray(a)

def design_notch (fc, Q, fs):
    """Calculates filter coefficients for a notch filter.

    Parameters
    ----------
    fc : float
        Center frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    fs :  float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    wc = 2 * np.pi * fc / fs
    wS = np.sin (wc)
    wC = np.cos (wc)
    alpha = wS / (2.0 * Q)

    a0 = 1.0 + alpha

    b = [1.0 / a0, -2.0 * wC / a0, 1.0 / a0]
    a = [1, -2.0 * wC / a0, (1.0 - alpha) / a0]
    return np.asarray (b), np.asarray(a)

def design_highshelf (fc, Q, gain, fs):
    """Calculates filter coefficients for a High Shelf filter.

    Parameters
    ----------
    fc : float
        Center frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    gain : float
        Linear gain for the shelved frequencies
    fs :  float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    A = np.sqrt (gain)
    wc = 2 * np.pi * fc / fs
    wS = np.sin (wc)
    wC = np.cos (wc)
    beta = np.sqrt (A) / Q

    a0 = ((A+1.0) - ((A-1.0) * wC) + (beta*wS))

    b = np.zeros (3)
    a = np.zeros (3)
    b[0] = A*((A+1.0) + ((A-1.0)*wC) + (beta*wS)) / a0
    b[1] = -2.0*A * ((A-1.0) + ((A+1.0)*wC)) / a0
    b[2] = A*((A+1.0) + ((A-1.0)*wC) - (beta*wS)) / a0

    a[0] = 1
    a[1] = 2.0 * ((A-1.0) - ((A+1.0)*wC)) / a0
    a[2] = ((A+1.0) - ((A-1.0)*wC)-(beta*wS)) / a0
    return b, a

def design_lowshelf (fc, Q, gain, fs):
    """Calculates filter coefficients for a Low Shelf filter.

    Parameters
    ----------
    fc : float
        Center frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    gain : float
        Linear gain for the shelved frequencies
    fs :  float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    A = np.sqrt (gain)
    wc = 2 * np.pi * fc / fs
    wS = np.sin (wc)
    wC = np.cos (wc)
    beta = np.sqrt (A) / Q

    a0 = ((A+1.0) + ((A-1.0) * wC) + (beta*wS))

    b = np.zeros (3)
    a = np.zeros (3)
    b[0] = A*((A+1.0) - ((A-1.0)*wC) + (beta*wS)) / a0
    b[1] = 2.0*A * ((A-1.0) - ((A+1.0)*wC)) / a0
    b[2] = A*((A+1.0) - ((A-1.0)*wC) - (beta*wS)) / a0

    a[0] = 1
    a[1] = -2.0 * ((A-1.0) + ((A+1.0)*wC)) / a0
    a[2] = ((A+1.0) + ((A-1.0)*wC)-(beta*wS)) / a0
    return b, a

def bilinear_biquad (b_s, a_s, fs, matchPole=False):
    # find freq to match with bilinear transform
    T = 1.0 / fs
    c = 2/T
    if (matchPole):
        wc = np.imag (np.roots (a_s))[0]
        c = wc / np.tan (wc * T / 2.0)
    c_2 = c*c

    # bilinear
    a = np.zeros (3)
    b = np.zeros (3)
    a0 = a_s[0] * c_2 + a_s[1] * c + a_s[2]

    a[0] = a0 / a0
    a[1] = 2.0 * (a_s[2] - a_s[0] * c_2) / a0
    a[2] = (a_s[0] * c_2 - a_s[1] * c + a_s[2]) / a0
    b[0] = (b_s[0] * c_2 + b_s[1] * c + b_s[2]) / a0
    b[1] = 2.0 * (b_s[2] - b_s[0] * c_2) / a0
    b[2] = (b_s[0] * c_2 - b_s[1] * c + b_s[2]) / a0

    return b, a
