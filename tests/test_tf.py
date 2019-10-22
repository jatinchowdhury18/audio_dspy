from unittest import TestCase
import numpy as np
import random as r
import scipy.signal as signal

import audio_dspy as adsp

_N_ = 1024
_tolerance_ = 0.15


class TestTFs(TestCase):
    def setUp(self):
        self.h = np.zeros(_N_)
        r.seed(0x1234)
        for n in range(_N_):
            self.h[n] = r.random() - 0.5

    def test_tf2linphase(self):
        htest = np.copy(self.h)
        h_lin = adsp.tf2linphase(htest)

        # test symmetry
        diffs = np.zeros(int(_N_/2))
        for n in range(int(_N_/2)):
            diffs[n] = np.abs(np.abs(h_lin[n+2]) - np.abs(h_lin[_N_-1-n]))
        self.assertTrue(np.max(diffs) < _tolerance_,
                        'Linear Phase IR is not symmetric! {}'.format(np.max(diffs)))

    def test_tf2minphase(self):
        htest = np.copy(self.h)
        h_min = adsp.tf2minphase(htest)

        # test group delay
        _, orig_delay = signal.group_delay((htest, 1))
        _, min_delay = signal.group_delay((h_min, 1))
        self.assertTrue(min_delay[0] <= orig_delay[0],
                        'Minimum phase IR does not have minimum group delay!')
