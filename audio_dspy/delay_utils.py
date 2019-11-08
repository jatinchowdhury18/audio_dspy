def delay_feedback_gain_for_t60(delay_samp, fs, t60):
    """
    Calculate the gain needed in a feedback delay line
    to achieve a desired T60

    Parameters
    ----------
    delay_samp : int
        The delay length in samples
    fs : float
        Sample rate
    t60 : float
        The desired T60 [seconds]

    Returns
    -------
    g : float
        The gain needed to achieve the desired T60 [linear gain]
    """
    n_times = t60 * fs / delay_samp
    return 0.001**(1/n_times)