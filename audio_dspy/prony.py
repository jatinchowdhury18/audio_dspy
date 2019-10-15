import numpy as np
import scipy.signal as signal
import scipy.linalg as linalg
import audio_dspy as adsp

def prony (x, nb, na):
    """Uses Prony's method to generate IIR filter coefficients
    that optimally match a given transfer function

    Parameters
    ----------
    x : ndarray
        Numpy array containing the transfer function
    nb : int
        Number of feedforward coefficients in the resulting filter
    na : int
        Number of feedback coefficients in the resulting filter

    Returns
    -------
    b : ndarray
        Feedforward coefficients
    a : ndarray
        Feedback coefficients
    """
    h = adsp.tf2minphase (x)
    k = len(h)-1
    H = np.mat (linalg.toeplitz (np.array(h), np.append([1], np.zeros(k))))
    H = H[:, 0:(na+1)]
    H1 = H[0:(nb+1), :]
    h1 = H[(nb+1):(k+1), 0]
    H2 = H[(nb+1):(k+1), 1:(na+1)]
    a = np.vstack((np.mat(1), -H2.I * h1))
    aT = a.T
    b = aT * H1.T
    
    return b.getA()[0], aT.getA()[0]

def allpass_warp (rho, h):
    """Performs allpass warping on a transfer function

    Parameters
    ----------
    rho : float
        Amount of warping to perform. On the range (-1, 1). Positive warping
        "expands" the spectrum, negative warping "shrinks"
    h : ndarray
        The transfer function to warp

    Returns
    -------
    h_warped : ndarray
        The warped transfer function
    """
    b_ap = np.array([rho, 1])
    a_ap = np.array([1, rho])

    x = np.zeros (len (h)); x[0] = 1
    h_warp = np.zeros (len (h))
    for n in range (len (h)):
        h_warp += h[n] * x
        x = signal.lfilter (b_ap, a_ap, x)
    return h_warp

def allpass_warp_roots (rho, b):
    """Performs allpass warping on a filter coefficients

    Parameters
    ----------
    rho : float
        Amount of warping to perform. On the range (-1, 1). Positive warping
        "expands" the spectrum, negative warping "shrinks"
    b : ndarray
        The filter coefficients

    Returns
    -------
    b_warped : ndarray
        The warped filter coefficients
    """
    roots = np.roots (b)
    warped_roots = np.zeros (len (roots), dtype=np.complex128)
    for n in range (len (roots)):
        mag = np.abs (roots[n])
        angle = np.angle (roots[n])
        warped_angle = np.arctan2 ((1 - rho**2) * np.sin (angle), (1 + rho**2) * np.cos (angle) - 2 * rho)
        warped_roots[n] = mag * np.exp (1j*warped_angle)
    return np.real (np.poly (warped_roots)) * b[0]

def prony_warped (x, nb, na, rho):
    """Uses Prony's method with frequency warping to generate IIR filter coefficients
    that optimally match a given transfer function

    Parameters
    ----------
    x : ndarray
        Numpy array containing the transfer function
    nb : int
        Number of feedforward coefficients in the resulting filter
    na : int
        Number of feedback coefficients in the resulting filter
    rho : float
        Amount of warping to perform. On the range (-1, 1). Positive warping
        "expands" the spectrum, negative warping "shrinks"

    Returns
    -------
    b : ndarray
        Feedforward coefficients
    a : ndarray
        Feedback coefficients
    """
    x = adsp.tf2minphase (x)
    x_warp = allpass_warp (rho, x)
    b_warped, a_warped = prony (x_warp, nb, na)
    b = allpass_warp_roots (-rho, b_warped)
    a = allpass_warp_roots (-rho, a_warped)
    return b, a
