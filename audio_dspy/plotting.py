import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patches
import audio_dspy as adsp


def plot_freqz_mag(w, H):
    """Plots the magnitude output of the scipy.signal.freqz function

    Parameters
    ----------
    w :  ndarray
        w output of freqz
    H : ndarray
        H output of freqz
    """
    plt.semilogx(w, 20 * np.log10(np.abs(H)))
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Frequency [Hz]')


def plot_freqz_angle(w, H):
    """Plots the phase output of the scipy.signal.freqz function

    Parameters
    ----------
    w :  ndarray
        w output of freqz
    H : ndarray
        H output of freqz
    """
    plt.semilogx(w, np.unwrap(np.angle(H)))
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [Hz]')


def plot_magnitude_response(b, a, worN=512, fs=2*np.pi):
    """Plots the magnitude response of a digital filter in dB, using second order sections

    Parameters
    ----------
    b: ndarray
        numerator (feed-forward) coefficients of the filter
    a: ndarray
        denominator (feed-backward) coefficients of the filter
    worN: {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is N=512).
        If an array_like, compute the response at the frequencies given. These are in the same units as fs.
    fs: float, optional
        sample rate of the filter
    """
    w, H = signal.freqz(b, a, worN=worN, fs=fs)
    plot_freqz_mag(w, H)


def plot_magnitude_response_sos(sos, worN=512, fs=2*np.pi):
    """Plots the magnitude response of a digital filter in dB

    Parameters
    ----------
    sos : array-like
        Filter to plot as a series of second-order sections
    worN: {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is N=512).
        If an array_like, compute the response at the frequencies given. These are in the same units as fs.
    fs: float, optional
        sample rate of the filter
    """
    w, H = signal.sosfreqz(sos, worN=worN, fs=fs)
    plot_freqz_mag(w, H)


def plot_phase_response(b, a, worN=512, fs=2*np.pi):
    """Plots the phase response of a digital filter in radians

    Parameters
    ----------
    b: ndarray
        numerator (feed-forward) coefficients of the filter
    a: ndarray
        denominator (feed-backward) coefficients of the filter
    worN: {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is N=512).
        If an array_like, compute the response at the frequencies given. These are in the same units as fs.
    fs: float, optional
        sample rate of the filter
    """
    w, H = signal.freqz(b, a, worN=worN, fs=fs)
    plot_freqz_angle(w, H)


def plot_phase_response_sos(sos, worN=512, fs=2*np.pi):
    """Plots the phase response of a digital filter in radians, using second order sections

    Parameters
    ----------
    sos : array-like
        Filter to plot as a series of second-order sections
    worN: {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is N=512).
        If an array_like, compute the response at the frequencies given. These are in the same units as fs.
    fs: float, optional
        sample rate of the filter
    """
    w, H = signal.sosfreqz(sos, worN=worN, fs=fs)
    plot_freqz_angle(w, H)


def plot_static_curve(function, range=10, num=1000):
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
    x = np.linspace(-range, range, num=num)
    y = function(x)
    plt.plot(x, y)
    plt.xlabel('Input Gain')
    plt.ylabel('Output Gain')


def plot_dynamic_curve(function, freq=100, fs=44100, range=10, num=1000):
    """Plots the dynamic curve of a nonlinear function at a specific frequency

    Parameters
    ----------
    function: lambda (float) : float
        function to plot the dynamic curve for
    freq: float, optional
        frequency [Hz] to plot the dynamic curve for, defaults to 100 Hz
    fs: float, optional
        sample rate [Hz] to use for the simulation, defaults to 44.1 kHz
    range: float, optional
        range on which to plot the dynamic curve [-range, range]
    num: int, optional
        number of points to plot
    """
    n = np.arange(num)
    x = range * np.sin(2 * np.pi * n * freq / fs)
    y = function(x)
    plt.plot(x, y)
    plt.xlabel('Input Gain')
    plt.ylabel('Output Gain')


def plot_harmonic_response(function, freq=100, fs=44100, gain=0.1, num=10000):
    """Plots the harmonic response of a nonlinear function at a specific frequency

    Parameters
    ----------
    function: lambda (float) : float
        function to plot the harmonic response for
    freq: float, optional
        frequency [Hz] to plot the harmonic response for, defaults to 100 Hz
    fs: float, optional
        sample rate [Hz] to use for the simulation, defaults to 44.1 kHz
    gain: float, optional
        gain to use for the input signal, defaults to 0.1
    num: int, optional
        number of points to plot
    """
    n = np.arange(num)
    x = gain*np.sin(2 * np.pi * n * freq / fs)
    y = function(np.copy(x))

    f = np.linspace(0, fs/2, num=num/2+1)
    H = adsp.normalize(np.fft.rfft(y))
    plt.semilogx(f, 20 * np.log10(np.abs(H)))
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')


def zplane(b, a):
    """Plots the pole-zero response of a digital filter

    Parameters
    ----------
    b : array-like
        feed-forward coefficients
    a : array-like
        feed-back coefficients
    """
    p = np.roots(a)
    z = np.roots(b)

    ax = plt.subplot(111)
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='white', ls='dashed')
    ax.add_patch(uc)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(z.real, z.imag, 'go', ms=10)
    plt.plot(p.real, p.imag, 'rx', ms=10)

    r = 1.5
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)
