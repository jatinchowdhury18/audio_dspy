from unittest import TestCase
import numpy as np
import scipy.signal as signal

import audio_dspy as adsp

import matplotlib.pyplot as plt

_fs_ = 44100


class TestFarina(TestCase):
    def setUp(self):
        g = adsp.delay_feedback_gain_for_t60(1, _fs_, 0.5)
        self.N = int(0.6 * _fs_)

        np.random.seed(0x2345)
        self.test_filt = adsp.normalize(
            np.random.randn(self.N) * g**np.arange(self.N))
        self.lin_sys = lambda sig: signal.convolve(sig, self.test_filt)
        self.nonlin_sys = lambda sig: signal.convolve(
            adsp.soft_clipper(1.1*sig, deg=9), self.test_filt)

    def test_inverse_probe_freq(self):
        far = adsp.Farina(10, _fs_)
        delta = signal.convolve(far.probe, far.inv_probe)

        freqs = np.fft.rfftfreq(len(delta), 1.0/_fs_)
        start = np.argwhere(freqs > 100)[0][0]
        end = np.argwhere(freqs < 16000)[-1][0]

        delta_fft = 20 * \
            np.log10(adsp.normalize(np.abs(np.fft.rfft(delta))))[start:end]
        rangeDB = np.max(delta_fft) - np.min(delta_fft)
        self.assertTrue(
            rangeDB < 2.0, 'Inverse probe does not have correct frequency response!')

    def test_inverse_probe_imp(self):
        far = adsp.Farina(10, _fs_)
        delta = signal.convolve(far.probe, far.inv_probe)
        delta_sort = np.sort(np.abs(delta))
        ratio = delta_sort[-1] / delta_sort[-2]

        self.assertTrue(
            ratio > 5, 'Probe and inverse probe does not convolve to a delta!')

    def test_freq_response_lin(self):
        far = adsp.Farina(20, _fs_)
        meas = self.lin_sys(far.probe)
        far.process_measurement(meas)

        freqs = np.fft.rfftfreq(self.N, 1.0/_fs_)
        start = np.argwhere(freqs > 100)[0][0]
        end = np.argwhere(freqs < 16000)[-1][0]

        test_fft = 20 * \
            np.log10(adsp.normalize(np.abs(np.fft.rfft(self.test_filt))))[
                start:end]
        far_fft = 20 * \
            np.log10(adsp.normalize(np.abs(np.fft.rfft(far.get_IR()[:self.N]))))[
                start:end]
        error = np.max(np.abs(test_fft - far_fft))

        self.assertTrue(
            error < 10, 'Incorrect frequency response for linear system!')

    def test_THD_lin(self):
        far = adsp.Farina(20, _fs_)
        meas = self.lin_sys(far.probe)
        far.process_measurement(meas)

        thd = far.getTHD()
        self.assertTrue(thd < 0.1, 'Incorrect THD for linear system!')

    def test_freq_response_nonlin(self):
        far = adsp.Farina(20, _fs_)
        meas = self.nonlin_sys(far.probe)
        far.process_measurement(meas)

        freqs = np.fft.rfftfreq(self.N, 1.0/_fs_)
        start = np.argwhere(freqs > 100)[0][0]
        end = np.argwhere(freqs < 16000)[-1][0]

        test_fft = 20 * \
            np.log10(adsp.normalize(np.abs(np.fft.rfft(self.test_filt))))[
                start:end]
        far_fft = 20 * \
            np.log10(adsp.normalize(np.abs(np.fft.rfft(far.get_IR()[:self.N]))))[
                start:end]
        error = np.max(np.abs(test_fft - far_fft))

        self.assertTrue(
            error < 10, 'Incorrect frequency response for nonlinear system!')

    def test_THD_nonlin(self):
        # get reference THD from sine wave test
        N = _fs_*2
        freq = 1000
        sine = np.sin(2 * np.pi * np.arange(N) * freq / _fs_)
        y = self.nonlin_sys(sine)

        V_rms = np.zeros(9)
        for n in range(1, 10):
            low_freq = freq*n - 30
            b_hpf, a_hpf = signal.butter(
                4, low_freq, btype='highpass', analog=False, fs=_fs_)

            high_freq = freq*n + 30
            b_lpf, a_lpf = signal.butter(
                4, high_freq, btype='lowpass', analog=False, fs=_fs_)

            harmonic = signal.lfilter(
                b_hpf, a_hpf, (signal.lfilter(b_lpf, a_lpf, y)))
            V_rms[n-1] = np.sqrt(np.mean(harmonic**2))

        sine_thd = np.sqrt(np.sum(V_rms[1:]**2)) / V_rms[0]

        # get measured THD from Farina
        far = adsp.Farina(20, _fs_)
        meas = self.nonlin_sys(far.probe)
        far.process_measurement(meas)
        far_thd = far.getTHD()

        self.assertTrue(True, f'Sine: {sine_thd}, Far: {far_thd}')
