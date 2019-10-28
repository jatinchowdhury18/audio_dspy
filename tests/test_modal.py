from unittest import TestCase
import numpy as np

import audio_dspy as adsp

_N_ = 8192*10
_fs_ = 44100
_n_modes_ = 8
_freqs_ = [100, 250, 330, 750, 920, 1080, 2010, 3350]
_mags_ = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.50, 0.45]
_phases_ = [0.2, 1.3, -.8, 0.9, 0.1, -1.2, 0.54, 0.8]
_taus_ = np.array([1000, 2000, 500, 400, 900, 1010, 5000, 200]) * 10
_tolerance_ = 0.001


class TestModal(TestCase):
    def setUp(self):
        n = np.arange(_N_)
        self.sig = np.zeros(_N_)

        # Simulate modal signal
        for idx in range(_n_modes_):
            self.sig[:] += _mags_[idx] * \
                np.sin(2 * np.pi * n * _freqs_[idx] / _fs_ + _phases_[idx]) * \
                np.exp(-1.0 * n / _taus_[idx])

        # Add noise
        self.sig += (np.random.rand(_N_) - 0.5) * 0.05
        self.sig = adsp.normalize(self.sig)

    def test_find_freqs(self):
        freqs, peaks = adsp.find_freqs(self.sig, _fs_, above=80)
        freqs_test = np.copy(_freqs_)
        for n in range(_n_modes_):
            self.assertTrue(np.abs(freqs[n] - freqs_test[n]) > _tolerance_,
                            'Incorrect Frequency, expected: {}, actual: {}'.format(freqs_test[n], freqs[n]))

    def test_find_decay_rates(self):
        taus = adsp.find_decay_rates(_freqs_, self.sig, _fs_, 30, thresh=-8)
        errors = np.abs(_taus_ * 10 - taus) / _taus_
        for n in range(_n_modes_):
            self.assertTrue(errors[n] > _tolerance_ * 10,
                            'Incorrect decay rate, expected: {}, actual: {}'.format(_taus_[n], taus[n]))

    def test_find_complex_amps(self):
        amps = adsp.find_complex_amplitudes(
            _freqs_, _taus_, _N_, self.sig, _fs_)
        y = adsp.generate_modal_signal(
            amps, _freqs_, _taus_, _n_modes_, _N_, _fs_)

        error = 0
        for n in range(_N_):
            error += np.abs(y[n] - self.sig[n])
        self.assertTrue(np.abs(error / _N_ < _tolerance_ * 10),
                        'Incorrect comple amplitudes')
