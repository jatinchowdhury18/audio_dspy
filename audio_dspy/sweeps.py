import numpy as np


def sweep_log(f0, f1, duration, fs):
    """Generates a logarithmic sine sweep

    Parameters
    ----------
    f0: float
        The frequency [Hz] at which to begin the sine sweep
    f1: float
        The frequency [Hz] at which to stop the sine sweep
    duration: float
        The length of time [seconds] over which to sweep the signal
    fs: float
        The sample rate [Hz]

    Returns
    -------
    x: ndarray
        A numpy array containing the sine sweep signal
    """
    N = int(duration * fs)
    n = np.arange(N)

    beta = N / np.log(f1 / f0)
    phase = 2 * np.pi * beta * f0 * (pow(f1 / f0, n / N) - 1.0)
    phi = np.pi / 180

    return np.cos((phase + phi)/fs)


def sweep_lin(duration, fs):
    """Generates a linear sine sweep

    Parameters
    ----------
    duration: float
        The length of time [seconds] over which to sweep the signal
    fs: float
        The sample rate [Hz]

    Returns
    -------
    x: ndarray
        A numpy array containing the sine sweep signal
    """
    N = int(duration * fs)
    n = np.arange(N)

    phase = 2 * np.pi * (((fs/2)/N) * n * n / 2)
    phi = np.pi / 180

    return np.cos((phase + phi)/fs)


def sweep2ir(dry_sweep, wet_sweep):
    """Converts a pair of input/output sine sweeps into an impulse response

    Parameters
    ----------
    dry_sweep: ndarray
        The dry sine sweep used as input to the system
    wet_sweep: ndarray
        The wet sine sweep, output of the system

    Returns
    -------
    h: ndarray
        The impulse response of the system
    """
    N = max(len(wet_sweep), len(dry_sweep))
    SS = np.fft.fft(dry_sweep, n=N)
    RS = np.fft.fft(wet_sweep, n=N)

    H = RS/SS
    h = np.real(np.fft.ifft(H))
    return h
