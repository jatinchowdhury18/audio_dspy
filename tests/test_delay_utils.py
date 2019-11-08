from unittest import TestCase
import numpy as np
import scipy.signal as signal

import audio_dspy as adsp

_tolerance_ = 0.001
_fs_ = 44100
_t60_ = 1  # second
_N_ = 2 * _fs_


class TestEQDesign(TestCase):
    def test_delay_feedback_gain(self):
        g = adsp.delay_feedback_gain_for_t60(1, _fs_, _t60_)

        x = adsp.impulse(_N_)
        for n in range(1, _N_):
            x[n] = x[n-1] * g

        t60_samp = int(_fs_ * _t60_)
        self.assertTrue(np.abs(x[t60_samp] - 0.001) < _tolerance_,
                        'Incorrect T60 gain! Expected: {}, Actual: {}'.format(0.001, x[t60_samp]))
