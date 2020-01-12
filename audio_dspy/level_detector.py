import numpy as np


def _calc_coefs(tau_ms, fs):
    """Calculate coefficients for exponential decay

    Parameters
    ----------
    tau_ms : float
        Time constant [ms]
    fs : float
        Sample rate [Hz]

    Returns
    -------
    b0, a1 : floats
        Coefficients
    """
    a1 = np.exp(-1.0 / (fs * tau_ms / 1000.0))
    b0 = 1.0 - a1
    return b0, a1


def _peak_process(x, a_coefs, r_coefs):
    """Perform level detection for peak detector

    Parameters
    ----------
    x : ndarray
        Input vector
    a_coefs : tuple
        Attack coefficients (b0, a1)
    r_coefs : tuple
        Release coefficients (b0, a1)

    Returns
    -------
    y : ndarray
        Output vector
    """
    y = np.copy(x)
    level_est = 0

    for n, x_samp in enumerate(x):
        if np.abs(x_samp) > level_est:  # attack mode
            level_est += a_coefs[0] * (np.abs(x_samp) - level_est)
        else:  # release mode
            level_est += r_coefs[0] * (np.abs(x_samp) - level_est)

        y[n] = level_est

    return y


def _rms_process(x, a_coefs, r_coefs):
    """Perform level detection for RMS detector

    Parameters
    ----------
    x : ndarray
        Input vector
    a_coefs : tuple
        Attack coefficients (b0, a1)
    r_coefs : tuple
        Release coefficients (b0, a1)

    Returns
    -------
    y : ndarray
        Output vector
    """
    y = np.copy(x)
    level_est = 0

    for n, x_samp in enumerate(x):
        x_in = x_samp**2
        if x_in > level_est:  # attack mode
            level_est = a_coefs[1] * level_est + a_coefs[0] * x_in
        else:  # release mode
            level_est = r_coefs[1] * level_est + r_coefs[0] * x_in

        y[n] = np.sqrt(level_est)

    return y


def _analog_process(x, a_coefs, r_coefs):
    """Perform level detection for analog detector

    Parameters
    ----------
    x : ndarray
        Input vector
    a_coefs : tuple
        Attack coefficients (b0, a1)
    r_coefs : tuple
        Release coefficients (b0, a1)

    Returns
    -------
    y : ndarray
        Output vector
    """
    y = np.copy(x)
    level_est = 0

    # "magic" constants
    alpha = 0.7722
    beta = 0.85872

    for n, x_samp in enumerate(x):
        rect = max(beta * (np.exp(alpha * x_samp) - 1.0), 0.0)
        if rect > level_est:  # attack mode
            level_est += a_coefs[0] * (rect - level_est)
        else:  # release mode
            level_est += r_coefs[0] * (rect - level_est)

        y[n] = level_est

    return y


def level_detect(x, fs, attack_ms=0.1, release_ms=100, mode='peak'):
    """Performs level detection on an input signal

    Parameters
    ----------
    x : ndarray
        Input vector
    fs : float
        Sample rate [Hz]
    attack_ms : float, optional
        Time constant for attack [ms]
    release_ms : float, optional
        Time constant for release [ms]
    mode : string, optional
        Type of detector. Should be one of:

            - 'peak' (peak detector, default)
            - 'rms' (rms detector)
            - 'analog' (analog style detector, based on detector circuit with Shockley diode)

    Returns
    -------
    y : ndarray
        Output vector
    """
    # Get coefficients
    b0_a, a1_a = _calc_coefs(attack_ms, fs)
    b0_r, a1_r = _calc_coefs(release_ms, fs)

    # form output vector
    if mode == 'analog':
        return _analog_process(x, (b0_a, a1_a), (b0_r, a1_r))

    if mode == 'rms':
        return _rms_process(x, (b0_a, a1_a), (b0_r, a1_r))

    # 'peak' (default)
    return _peak_process(x, (b0_a, a1_a), (b0_r, a1_r))
