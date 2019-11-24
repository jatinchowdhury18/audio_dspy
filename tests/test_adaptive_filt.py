from unittest import TestCase
import numpy as np
import scipy.signal as signal
import random as r

import audio_dspy as adsp

_N_ = 4096
_fs_ = 44100
_L_ = 128
_tolerance_ = 0.0001


class TestAdaptiveFilts(TestCase):
    def setUp(self):
        r.seed(0x4567)

        self.x = np.zeros(_N_)
        for n in range(_N_):
            self.x[n] = r.random() - 0.5

        self.w = np.zeros(_L_)
        for n in range(_L_):
            self.w[n] = r.random() - 0.5

        self.y = signal.lfilter(self.w, [1], self.x)[:_N_]

    def test_LMS(self):
        y, e, w = adsp.LMS(self.x, self.y, 0.1, _L_)
        error = np.max(e[int(_N_*3/4):]**2)
        self.assertTrue(
            error < _tolerance_, 'Too much error in prediction! Max error: {}'.format(error))

    def test_NLMS(self):
        y, e, w = adsp.NLMS(self.x, self.y, 0.1, _L_)
        error = np.max(e[int(_N_*3/4):]**2)
        self.assertTrue(
            error < _tolerance_, 'Too much error in prediction! Max error: {}'.format(error))

    def test_NLLMS(self):
        y, e, w = adsp.NL_LMS(self.x, self.y, 0.1, _L_,
                              lambda x: np.tanh(x), lambda x: 1.0/np.cosh(x)**2)
        error = np.max(e[int(_N_*3/4):]**2) / \
            1000000  # @TODO: improvie this error
        self.assertTrue(
            error < _tolerance_, 'Too much error in prediction! Max error: {}'.format(error))
