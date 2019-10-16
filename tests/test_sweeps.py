from unittest import TestCase
import numpy as np

import audio_dspy as adsp

_fs_ = 44100
_tolrance_ = 0.00001

class TestSweeps(TestCase):
    def test_log_sweep(self):
        sweep = adsp.sweep_log (1, _fs_/2, 10, _fs_)
        sweep2 = adsp.sweep_log (1, _fs_/2, 10, _fs_)
        h = adsp.sweep2ir (sweep, sweep2)
        self.assertTrue (self.diff_vs_imp (h) < _tolrance_, 'Log Sweep response is not flat!')

    def test_lin_sweep(self):
        sweep = adsp.sweep_lin (10, _fs_)
        sweep2 = adsp.sweep_lin (10, _fs_)
        h = adsp.sweep2ir (sweep, sweep2)
        self.assertTrue (self.diff_vs_imp (h) < _tolrance_, 'Log Sweep response is not flat!')

    def diff_vs_imp(self, h):
        test_h = np.zeros (len (h)); test_h[0] = 1
        diff = 0
        for n in range (len (h)):
            diff += np.abs (h[n] - test_h[n])
        return diff
        
        
