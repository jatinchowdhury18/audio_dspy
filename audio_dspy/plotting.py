import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import audio_dspy as adsp


def plot_freqz_mag(w, H, norm=False):
    """Plots the magnitude output of the scipy.signal.freqz function

    Parameters
    ----------
    w :  ndarray
        w output of freqz
    H : ndarray
        H output of freqz
    norm : bool, optional
        Should normalize the magnitude response
    """
    if norm:
        H = adsp.normalize(H)

    plt.semilogx(w, 20 * np.log10(np.abs(H)))
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Frequency [Hz]')


def plot_freqz_angle(w, H, log=True):
    """Plots the phase output of the scipy.signal.freqz function

    Parameters
    ----------
    w :  ndarray
        w output of freqz
    H : ndarray
        H output of freqz
    log : bool, optional
        Should plot log scale
    """
    if log:
        plt.semilogx(w, np.unwrap(np.angle(H)))
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        plt.plot(w, np.unwrap(np.angle(H)))
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [Hz]')


def plot_magnitude_response(b, a, worN=512, fs=2*np.pi, norm=False):
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
    norm : bool, optional
        Should normalize the magnitude response
    """
    w, H = signal.freqz(b, a, worN=worN, fs=fs)
    plot_freqz_mag(w, H, norm=norm)


def plot_magnitude_response_sos(sos, worN=512, fs=2*np.pi, norm=False):
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
    norm : bool, optional
        Should normalize the magnitude response
    """
    w, H = signal.sosfreqz(sos, worN=worN, fs=fs)
    plot_freqz_mag(w, H, norm=norm)


def plot_phase_response(b, a, worN=512, fs=2*np.pi, log=True):
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
    log : bool, optional
        Should plot log scale
    """
    w, H = signal.freqz(b, a, worN=worN, fs=fs)
    plot_freqz_angle(w, H, log=log)


def plot_phase_response_sos(sos, worN=512, fs=2*np.pi, log=True):
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
    log : bool, optional
        Should plot log scale
    """
    w, H = signal.sosfreqz(sos, worN=worN, fs=fs)
    plot_freqz_angle(w, H, log=log)


def plot_static_curve(function, gain=10, num=1000):
    """Plots the static curve of a nonlinear function

    Parameters
    ----------
    function: lambda (float) : float
        function to plot the static curve for
    gain: float, optional
        range of gains on which to plot the static curve [-gain, gain]
    num: int, optional
        number of points to plot
    """
    x = np.linspace(-gain, gain, num=num)
    y = function(x)
    plt.plot(x, y)
    plt.xlabel('Input Gain')
    plt.ylabel('Output Gain')


def plot_dynamic_curve(function, freq=100, fs=44100, gain=10, num=1000):
    """Plots the dynamic curve of a nonlinear function at a specific frequency

    Parameters
    ----------
    function: lambda (float) : float
        function to plot the dynamic curve for
    freq: float, optional
        frequency [Hz] to plot the dynamic curve for, defaults to 100 Hz
    fs: float, optional
        sample rate [Hz] to use for the simulation, defaults to 44.1 kHz
    gain: float, optional
        range of gains on which to plot the dynamic curve [-gain, gain]
    num: int, optional
        number of points to plot
    """
    n = np.arange(num)
    x = gain * np.sin(2 * np.pi * n * freq / fs)
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


def zplane(b, a, radius=1.5):
    """Plots the pole-zero response of a digital filter

    Parameters
    ----------
    b : array-like
        feed-forward coefficients
    a : array-like
        feed-back coefficients
    radius : float
        The radius to plot for (default 1.5)
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

    plt.axis('scaled')
    plt.axis([-radius, radius, -radius, radius])

    ticks = []
    for n in np.linspace(np.max([radius-0.5, 1]), 0, endpoint=False, num=2):
        ticks.append(n)
        ticks.append(-n)
    plt.xticks(ticks)
    plt.yticks(ticks)


def plot_spectrogram(x, fs, win_size=1024, dbRange=180, title=''):
    """ Plots a dB spectrogram of the input signal and takes care of most of the formatting to get a standard log frequency scale spectrogram.

    Parameters
    ----------
    x : array-like
        Signal to plot the spectrogram of
    fs : float
        Sample rate of the signal
    win_size : int, optional
        Window size to use (default 1024)
    dbRange : float, optional
        The range of Decibels to include in the spectrogram (default 180)
    title : string, optional
        The title to use for the figure
    """
    fig = plt.figure()
    gs = GridSpec(7, 1)

    # Plot the time domain signal on top
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(np.arange(0., len(x))/fs, x, '-')
    ax1.grid()
    ax1.set(ylabel='Amplitude', title=title)
    ax1.set_ylim([-1., 1.])
    ax1.set_xlim([0, len(x)/fs])

    ax2 = fig.add_subplot(gs[2:7, 0])
    Pxx, freqs, t, _ = plt.specgram(x, NFFT=win_size, Fs=fs, window=np.hamming(
        win_size), noverlap=win_size/2, pad_to=win_size*2)
    Pxx_dB = 20*np.log10(Pxx/np.max(Pxx))
    Pxx_dB = np.where(Pxx_dB < -dbRange, -dbRange, Pxx_dB)

    cores = ax2.pcolormesh(
        t, freqs, Pxx_dB, cmap='inferno', vmin=-dbRange, vmax=0)
    ax2.set(ylabel='Frequency (Hz)', xlabel='Time (s)')

    # Add a colour bar
    cbar = plt.colorbar(cores, orientation="horizontal", pad=0.25)
    cbar.set_label('Power (dB)')
    cbar.set_clim(-dbRange, 0)
    ax2.set_yscale('log')
    ax2.set_ylim([25., fs/2])
    ax2.set_xlim([0, len(x)/fs])

    plt.show()
