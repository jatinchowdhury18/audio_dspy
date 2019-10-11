import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.ticker

def plot_magnitude_response (b, a, fs=2*np.pi):
    """Plots the magnitude response of a digital filter in dB

    Parameters
    ----------
    b: ndarray
        numerator (feed-forward) coefficients of the filter
    a: ndarray
        denominator (feed-backward) coefficients of the filter
    fs: float, optional
        sample rate of the filter
    """
    w, H = signal.freqz (b, a, fs=fs)
    plt.semilogx (w, 20 * np.log10 (np.abs (H)))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel ('Magnitude [dB]')
    plt.xlabel ('Frequency [Hz]')

def plot_phase_response (b, a, fs=2*np.pi):
    """Plots the phase response of a digital filter in radians

    Parameters
    ----------
    b: ndarray
        numerator (feed-forward) coefficients of the filter
    a: ndarray
        denominator (feed-backward) coefficients of the filter
    fs: float, optional
        sample rate of the filter
    """
    w, H = signal.freqz (b, a, fs=fs)
    plt.semilogx (w, np.unwrap (np.angle (H)))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel ('Phase [rad]')
    plt.xlabel ('Frequency [Hz]')

def plot_static_curve (function, range=10, num=1000):
    """Plots the static curve of a nonlinear function

    Parameters
    ----------
    function: lambda (float) : float
        function to plot the static curve for
    range: float, optional
        range on which to plot the static curve
    num: int, optional
        number of points to plot
    """
    x = np.linspace (-range, range, num=num)
    y = function (x)
    plt.plot (x, y)
    plt.xlabel ('Input Gain')
    plt.ylabel ('Output Gain')

def plot_dynamic_curve (function, freq=100, fs=44100, range=10, num=1000):
    """Plots the static curve of a nonlinear function

    Parameters
    ----------
    function: lambda (float) : float
        function to plot the static curve for
    range: float, optional
        range on which to plot the static curve [-range, range]
    num: int, optional
        number of points to plot
    """
    n = np.arange (num)
    x = range * np.sin (2 * np.pi * n * freq / fs)
    y = function (x)
    plt.plot (x, y)
    plt.xlabel ('Input Gain')
    plt.ylabel ('Output Gain')
