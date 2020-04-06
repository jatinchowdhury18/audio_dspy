import numpy as np
import audio_dspy as adsp
import scipy.signal as signal
from functools import wraps


class Farina:
    """
    Class that implements Alberto Farina's method [1]_ for
    simultaneously measuring frequency response and harmonic
    distortion of weakly nonlinear systems.

    References
    ----------
    .. [1] A. Farina "Simultaneous Measurement of Impulse Response
           and Distortion with a Swept-Sine Technique", Audio
           Engineering Society Convention 108, Feb. 2000
    """

    def __init__(self, duration, fs, f0=20, f1=20000):
        """
        Creates an object to create and process a Farina-style measurement.

        Parameters
        ----------
        duration : float
            length [seconds] of the desired measurement signal
        fs : float
            sample rate [Hz] of the measurement signal
        f0 : float
            Frequency [Hz] at which to start the measurement
        f1 : float
            Frequency [Hz] at which to end the measurement
        """
        N = int(duration * fs)
        self.fs = fs

        # create probe and inverse probe
        self.probe = adsp.sweep_log(f0, f1, duration, fs)
        R = np.log(f1 / f0)
        k = np.exp(np.arange(N) * R / N)
        self.inv_probe = np.flip(self.probe) / k

        # @TEST: test that probe convolved with inverse has flat spectrum,
        # and impulse-like response

        # determin times to look for harmonics
        self.far_response = None
        self.harm_times = [0]
        mult = 1
        while True:
            mult += 1
            delta_n = int(N * np.log(mult) / np.log(f1/f0))
            self.harm_times.append(delta_n)
            if self.harm_times[-1] - self.harm_times[-2] < fs * 0.05:
                break

    def process_measurement(self, measurement):
        """
        Processes a measurement made using the probe signal
        for this object.
        """
        self.far_response = adsp.normalize(
            signal.convolve(measurement, self.inv_probe))

        amax = np.argmax(self.far_response)
        level = adsp.level_detect(self.far_response, self.fs)
        off = int(self.fs/10)
        amin = np.argwhere(level[amax-off:amax] < 0.05)[-1][0]
        amax = amax - (off - amin)
        end = amax + np.argwhere(level[amax:] < 10**(-60/20))[0][0]

        self.harm_responses = [self.far_response[amax:end]]
        for i in range(1, len(self.harm_times)):
            start = amax - self.harm_times[i]
            end = amax - self.harm_times[i-1]
            self.harm_responses.append(self.far_response[start:end])

    def _check_meas(func):
        """
        Decorator to make sure the measurement has been
        processed before attempting to access anything that
        depends on it.
        """
        @wraps(func)
        def checker(self, *args, **kwargs):
            if self.far_response is None:
                assert False, 'You must process a measurement before calling this function'
            return func(self, *args, **kwargs)
        return checker

    @_check_meas
    def get_harm_response(self, harm_num):
        """
        Returns the impulse response for a certain harmonic
        of the system. Note that the fundamental is the 1st harmonic.
        """
        assert harm_num > 0, 'Harmonic number must be greater than zero!'
        assert harm_num < len(self.harm_times), 'Harmonic number too large!'
        return self.harm_responses[harm_num-1]

    @_check_meas
    def get_IR(self):
        """
        Returns the impulse response for the linear
        part of the system.
        """
        return self.get_harm_response(1)

    @_check_meas
    def getTHD(self):
        """
        Returns the estimated total harmonic distortion for the system.
        """
        rms_vals = np.zeros(len(self.harm_responses))
        for idx, response in enumerate(self.harm_responses):
            rms_vals[idx] = np.sqrt(np.mean(response**2))

        return np.sqrt(np.sum(rms_vals[1:]**2)) / rms_vals[0]
