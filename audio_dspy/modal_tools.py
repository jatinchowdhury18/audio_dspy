import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import audio_dspy as adsp


def find_freqs(x, fs, thresh=30, above=0, frac_off=0.1, plot=False):
    """
    Find the mode frequencies of a signal

    Parameters
    ----------
    x : ndarray
        signal to analyze
    fs : float
        sample rate of the signal
    thresh : float, optional
        threshold to use for finding modes [dB]
    above : float, optional
        lower limit frequency to look for modes
    frac_off : float, optional
        to avoid finding multiple peaks for the same mode,
        this parameter defines a fractional offset for
        frequency breaks between modes
    plot : bool, optional
        should plot this analysis

    Returns
    -------
    freqs : ndarray
        Mode frequencies [Hz]
    peaks : ndarray
        Mode magnitudes [gain]
    """
    X = np.fft.rfft(x)
    f = np.linspace(0, fs/2, num=len(X))

    X_freqs = []
    X_peaks = []
    for k in range(len(X)):
        # check if above thresh
        Mag = 20 * np.log10(np.abs(X[k]))
        if (Mag < thresh):
            continue

        # check if within frac_off of a larger peak
        k_test = k
        flag = 0
        while(f[k_test] < f[k] * (1 + frac_off)):
            k_test += 1
            if 20*np.log10(np.abs(X[k_test])) > Mag:
                flag = 1
                break

        k_test = k
        while(f[k_test] > f[k] * (1 - frac_off)):
            k_test -= 1
            if 20*np.log10(np.abs(X[k_test])) > Mag:
                flag = 1
                break

        if flag == 1:
            continue

        # if above lower limit
        if (f[k] > above):
            X_freqs.append(f[k])
            X_peaks.append(np.abs(X[k]))

    # plot if needed
    if plot:
        plt.semilogx(f, 20 * np.log10(np.abs(X)))
        for freq in X_freqs:
            plt.axvline(freq, color='r')
        plt.ylabel('Magnitude [dB]')
        plt.xlabel('Frequency [Hz]')

    return np.asarray(X_freqs), np.asarray(X_peaks)


def energy_envelope(sig, fs, eta=0.01):
    """
    Find the energy envelope of a signal

    Parameters
    ----------
    sig : ndarray
        Signal to analyze
    fs : float
        Sample rate of the signal
    eta : float, optional
        Speed of the envelope filter

    Returns
    -------
    envelope : ndarray
        The envelop of the signal
    """
    envelope = np.zeros(len(sig))

    z1 = 0
    a1 = np.exp(-1 / (eta * fs))
    b0 = 1-a1
    for n in range(len(sig)):
        x0 = sig[n]*sig[n]
        y0 = a1 * z1 + b0 * x0

        z1 = y0
        envelope[n] = np.sqrt(y0)

    return envelope


def filt_mode(x, freq, fs, width, order=4):
    """
    Filter the signal around a mode frequency

    Parameters
    ----------
    x : ndarray
        The original signal
    freq : float
        The mode frequency to filter out
    fs : float
        The sample rate of the signal
    width : float
        The width of frequencies around the mode to filter
    order : int, optional
        The order of filter to use

    Returns
    -------
    x_filt : ndarray
        The signal filtered around the mode frequency
    """
    lowFreq = freq - width/2
    b_hpf = np.array([1.0, 0.0, 0.0])
    a_hpf = np.array([1.0, 0.0, 0.0])
    if lowFreq > 0.1:
        Wn = lowFreq / (fs/2)
        b_hpf, a_hpf = signal.butter(order, Wn, btype='highpass', analog=False)

    highFreq = freq + width/2
    b_lpf = np.array([1.0, 0.0, 0.0])
    a_lpf = np.array([1.0, 0.0, 0.0])
    if highFreq < fs/2:
        Wn = highFreq / (fs/2)
        b_lpf, a_lpf = signal.butter(order, Wn, btype='lowpass', analog=False)

    x_filt = adsp.normalize(signal.lfilter(
        b_hpf, a_hpf, (signal.lfilter(b_lpf, a_lpf, x))))
    return x_filt


def find_decay_rates(freqs, x, fs, filt_width, thresh=-60, eta=0.01, plot=False):
    """
    Find the decay rate of a set of modes

    Parameters
    ----------
    freqs : ndarray
        The mode frequencies
    x : ndarray
        The original signal
    fs : float
        Sample rate
    filt_width : float
        The range of frequencies to filter about each mode
    thresh : float, optional
        The threshold at which to stop fitting the decay rate [dB]
    eta : float, optional
        The speed of the filter to use to find the energy envelope of the mode
    plot : bool, optiona;
        Should plot the decay rate model for each mode

    Returns
    -------
    taus : ndarray
        The decay rates in units [gain/sample]
    """
    taus = []
    for freq in freqs:
        x_filt = filt_mode(x, freq, fs, filt_width)
        env = adsp.normalize(energy_envelope(x_filt, fs, eta))

        start = int(np.argwhere(20 * np.log10(env) > -1)[0])
        end = int(np.argwhere(20 * np.log10(env[start:]) < thresh)[0])
        slope, _, _, _, _ = stats.linregress(
            np.arange(len(env[start:end])), 20 * np.log10(env[start:end]))

        gamma = 10**(slope/20)
        tau = -1 / np.log(gamma)
        taus.append(tau)

        if plot:
            plt.figure()
            plt.title('Decay model for mode = {0:.2f} Hz'.format(freq))
            n = np.arange(len(env))
            plt.plot(n / fs, 20*np.log10(x_filt))
            plt.plot(n / fs, 20*np.log10(env))
            plt.plot(n / fs, 20*np.log10(np.exp(-1.0 * n / tau)), color='r')
            plt.xlabel('Time [s]')
            plt.ylim(thresh * 2, 5)

    return np.asarray(taus)


def find_complex_amplitudes(freqs, taus, T, x, fs):
    """
    Find optimal complex amplitudes for the modal frequencies using least squares.

    Parameters
    ----------
    freqs : ndarray
        Mode frequencies [Hz]
    taus : ndarray
        Mode decay rates [gain/sample]
    T : int
        Length of the time vector [samples] to use for optimization
    x : ndarray
        The time domain signal to optimize for
    fs : float
        The sampel rate of the time domain signal

    Returns
    -------
    amps : ndarray
        The complex amplitudes of the modes
    """
    num_modes = len(freqs)
    M = np.zeros((T, num_modes), dtype=np.complex128)
    for n in range(T):
        for m in range(num_modes):
            M[n][m] = np.exp((-1.0/taus[m] + 1j*(2*np.pi*freqs[m]/fs))*n)

    a, _, _, _ = np.linalg.lstsq(M, x[:T], rcond=None)
    return a


def design_modal_filter(amp, freq, tau, fs):
    """
    Designs a modal filter for a modal model

    Parameters
    ----------
    amp : complex float
        Complex amplitude of the mode
    freq : float
        Frequency of the mode [Hz]
    tau : float
        Decay rate of the mode [gain/sample]
    fs : float
        Sample rate

    Returns
    -------
    b : ndarray
        Feed-forward filter coefficients
    a : ndarray
        Feed-back filter coefficients
    """
    b = np.array([2.0*np.real(amp), -2.0 * np.exp(-1.0 / tau) *
                  np.real(amp * np.exp(-1j * (2 * np.pi * freq / fs))), 0.0])
    a = np.array([1.0, -2.0 * np.exp(-1.0 / tau) *
                  np.cos(2 * np.pi * freq / fs), np.exp(-2.0 / tau)])
    return b, a


def generate_modal_signal(amps, freqs, taus, num_modes, N, fs):
    """
    Generates a modal signal from modal model information

    Parameters
    ----------
    amps : array-like
        The complex amplitudes of the modes
    freqs : array-like
        The frequencies of the modes [Hz]
    taus : array-like
        The decay rates of the modes [gain/sample]
    num_modes : int
        The number of modes
    N : int
        The length of the signal to generates [samples]
    fs : float
        The sample rate
    """
    assert num_modes == len(amps), 'Incorrect number of amplitudes'
    assert num_modes == len(freqs), 'Incorrect number of frequencies'
    assert num_modes == len(taus), 'Incorrect number of decay rates'

    filts = []
    for n in range(num_modes):
        b, a = design_modal_filter(amps[n], freqs[n], taus[n], fs)
        filts.append([b, a])

    imp = adsp.impulse(N)
    y = np.zeros(N)

    for n in range(num_modes):
        b = filts[n][0]
        a = filts[n][1]

        y += signal.lfilter(b, a, imp)

    return adsp.normalize(y)
