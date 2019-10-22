import numpy as np
import scipy.signal as signal


def design_bell(fc, Q, gain, fs):
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
    c = 1.0 / np.tan(wc / 2.0)
    phi = c*c
    Knum = c / Q
    Kdenom = Knum

    if (gain > 1.0):
        Knum *= gain
    elif (gain < 1.0):
        Kdenom /= gain

    a0 = phi + Kdenom + 1.0

    b = [(phi + Knum + 1.0) / a0, 2.0 *
         (1.0 - phi) / a0, (phi - Knum + 1.0) / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - Kdenom + 1.0) / a0]

    return np.asarray(b), np.asarray(a)


def add_to_sos(sos, b, a):
    """Add a new filter to a set of second order sections

    Parameters
    ----------
    sos : array-like
        Set of second order sections
    b : array-like
        feed-forward coefficients of filter to add
    a : array-like
        feed-back coefficients of filter to add

    Returns
    -------
    sos : array-like
        New set of second order sections
    """
    z, p, k = signal.tf2zpk(b, a)
    if (np.size(sos) == 0):
        sos = signal.zpk2sos(z, p, k)
    else:
        sos = np.append(sos, signal.zpk2sos(z, p, k), axis=0)
    return sos


def butter_Qs(n):
    """Generate Q-values for an n-th order Butterworth filter

    Parameters:
    n : int
        order of filter to generate Q values for

    Returns
    -------
    q_values : array-like
        Set of Q-values for this order filter
    """
    k = 1
    lim = int(n / 2)
    Qs = []

    while k <= lim:
        b = -2 * np.cos((2*k + n - 1) * np.pi / (2*n))
        Qs.append(1/b)
        k += 1

    return np.flip(np.asarray(Qs))


def design_LPF1(fc, fs):
    """Calculates filter coefficients for a 1st-order lowpass filter

    Parameters
    ----------
    fc : float
        Cutoff frequency of the filter in Hz
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
    c = 1.0 / np.tan(wc / 2.0)
    a0 = c + 1.0

    b = [1 / a0, 1.0 / a0]
    a = [1, (1.0 - c) / a0]

    return np.asarray(b), np.asarray(a)


def design_LPF2(fc, Q,  fs):
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
    c = 1.0 / np.tan(wc / 2.0)
    phi = c*c
    K = c / Q
    a0 = phi + K + 1.0

    b = [1 / a0, 2.0 / a0, 1.0 / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - K + 1.0) / a0]

    return np.asarray(b), np.asarray(a)


def design_LPFN(fc, Q, N, fs):
    """Calculates filter coefficients for a Nth-order lowpass filter

    Parameters
    ----------
    fc : float
        Cutoff frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    N :  int
        Desired filter order
    fs : float
        Sample rate in Hz

    Returns
    -------
    sos : ndarray
        Filter coefficients as a set of second-order sections
    """
    sos = np.array([[]])
    filterOrd = N
    butterQs = butter_Qs(N)
    while (N - 2 >= 0):
        thisQ = butterQs[int(N/2) - 1]
        if (N == filterOrd):
            thisQ *= Q / 0.7071

        b, a = design_LPF2(fc, thisQ, fs)
        sos = add_to_sos(sos, b, a)
        N -= 2

    if (N > 0):
        b, a = design_LPF1(fc, fs)
        sos = add_to_sos(sos, b, a)

    return sos


def design_HPF1(fc, fs):
    """Calculates filter coefficients for a 1st-order highpass filter

    Parameters
    ----------
    fc : float
        Cutoff frequency of the filter in Hz
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
    c = 1.0 / np.tan(wc / 2.0)
    a0 = c + 1.0

    b = [c / a0, -c / a0]
    a = [1, (1.0 - c) / a0]

    return np.asarray(b), np.asarray(a)


def design_HPF2(fc, Q,  fs):
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
    c = 1.0 / np.tan(wc / 2.0)
    phi = c*c
    K = c / Q
    a0 = phi + K + 1.0

    b = [phi / a0, -2.0 * phi / a0, phi / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - K + 1.0) / a0]
    return np.asarray(b), np.asarray(a)


def design_HPFN(fc, Q, N, fs):
    """Calculates filter coefficients for a Nth-order highpass filter

    Parameters
    ----------
    fc : float
        Cutoff frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    N :  int
        Desired filter order
    fs : float
        Sample rate in Hz

    Returns
    -------
    sos : ndarray
        Filter coefficients as a set of second-order sections
    """
    sos = np.array([[]])
    filterOrd = N
    butterQs = butter_Qs(N)
    while (N - 2 >= 0):
        thisQ = butterQs[int(N/2) - 1]
        if (N == filterOrd):
            thisQ *= Q / 0.7071

        b, a = design_HPF2(fc, thisQ, fs)
        sos = add_to_sos(sos, b, a)
        N -= 2

    if (N > 0):
        b, a = design_HPF1(fc, fs)
        sos = add_to_sos(sos, b, a)

    return sos


def design_notch(fc, Q, fs):
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
    wS = np.sin(wc)
    wC = np.cos(wc)
    alpha = wS / (2.0 * Q)

    a0 = 1.0 + alpha

    b = [1.0 / a0, -2.0 * wC / a0, 1.0 / a0]
    a = [1, -2.0 * wC / a0, (1.0 - alpha) / a0]
    return np.asarray(b), np.asarray(a)


def design_highshelf(fc, Q, gain, fs):
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
    A = np.sqrt(gain)
    wc = 2 * np.pi * fc / fs
    wS = np.sin(wc)
    wC = np.cos(wc)
    beta = np.sqrt(A) / Q

    a0 = ((A+1.0) - ((A-1.0) * wC) + (beta*wS))

    b = np.zeros(3)
    a = np.zeros(3)
    b[0] = A*((A+1.0) + ((A-1.0)*wC) + (beta*wS)) / a0
    b[1] = -2.0*A * ((A-1.0) + ((A+1.0)*wC)) / a0
    b[2] = A*((A+1.0) + ((A-1.0)*wC) - (beta*wS)) / a0

    a[0] = 1
    a[1] = 2.0 * ((A-1.0) - ((A+1.0)*wC)) / a0
    a[2] = ((A+1.0) - ((A-1.0)*wC)-(beta*wS)) / a0
    return b, a


def design_lowshelf(fc, Q, gain, fs):
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
    A = np.sqrt(gain)
    wc = 2 * np.pi * fc / fs
    wS = np.sin(wc)
    wC = np.cos(wc)
    beta = np.sqrt(A) / Q

    a0 = ((A+1.0) + ((A-1.0) * wC) + (beta*wS))

    b = np.zeros(3)
    a = np.zeros(3)
    b[0] = A*((A+1.0) - ((A-1.0)*wC) + (beta*wS)) / a0
    b[1] = 2.0*A * ((A-1.0) - ((A+1.0)*wC)) / a0
    b[2] = A*((A+1.0) - ((A-1.0)*wC) - (beta*wS)) / a0

    a[0] = 1
    a[1] = -2.0 * ((A-1.0) + ((A+1.0)*wC)) / a0
    a[2] = ((A+1.0) + ((A-1.0)*wC)-(beta*wS)) / a0
    return b, a


def bilinear_biquad(b_s, a_s, fs, matchPole=False):
    # find freq to match with bilinear transform
    T = 1.0 / fs
    c = 2/T
    if (matchPole):
        wc = np.imag(np.roots(a_s))[0]
        c = wc / np.tan(wc * T / 2.0)
    c_2 = c*c

    # bilinear
    a = np.zeros(3)
    b = np.zeros(3)
    a0 = a_s[0] * c_2 + a_s[1] * c + a_s[2]

    a[0] = a0 / a0
    a[1] = 2.0 * (a_s[2] - a_s[0] * c_2) / a0
    a[2] = (a_s[0] * c_2 - a_s[1] * c + a_s[2]) / a0
    b[0] = (b_s[0] * c_2 + b_s[1] * c + b_s[2]) / a0
    b[1] = 2.0 * (b_s[2] - b_s[0] * c_2) / a0
    b[2] = (b_s[0] * c_2 - b_s[1] * c + b_s[2]) / a0

    return b, a
