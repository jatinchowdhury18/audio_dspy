from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

import audio_dspy as adsp

_N_ = 30000
_fs_ = 44100
_tolerance_ = 0.001
_points_ = [5000, 15000, 25000]
_exps_ = [0.0, 1.0, 0.0]


class TestLevelDetector(TestCase):
    def setUp(self):
        self.sig = np.zeros(_N_)
        self.sig[10000:20000] = 1  # unit step

        self.sig_att = np.zeros(_N_)
        self.sig_att[15000:] = 1

        self.sig_rel = np.zeros(_N_)
        self.sig_rel[:15000] = 1

    def check_point(self, y, point, exp, mode):
        self.assertTrue(np.abs(y[point] - exp) < _tolerance_,
                        'Expected: {}, Actual: {}, Point: {}, Mode: {}'.format(exp, y[point], point, mode))

    def run_mode(self, mode):
        y = adsp.level_detect(self.sig, _fs_, release_ms=0.1, mode=mode)
        for n, p in enumerate(_points_):
            self.check_point(y, p, _exps_[n], mode)

    def test_modes(self):
        for mode in ['peak', 'rms', 'analog']:
            self.run_mode(mode)

    def test_attack(self):
        y_slow = adsp.level_detect(self.sig_att, _fs_, attack_ms=20)
        speed_slow = 1 / np.argmax(y_slow > (1.0 - _tolerance_))

        y_fast = adsp.level_detect(self.sig_att, _fs_, attack_ms=0.5)
        speed_fast = 1 / np.argmax(y_fast > (1.0 - _tolerance_))

        self.assertTrue(speed_fast > speed_slow, 'Fast isn\'t faster than slow!, speed_fast: {}, speed_slow {}'.format(
            speed_fast, speed_slow))

    def test_release(self):
        y_slow = adsp.level_detect(self.sig_rel, _fs_, release_ms=20)
        speed_slow = 1 / np.argmax(y_slow < (0.0 + _tolerance_))

        y_fast = adsp.level_detect(self.sig_rel, _fs_, release_ms=5)
        speed_fast = 1 / np.argmax(y_fast < (0.0 + _tolerance_))

        self.assertTrue(speed_fast > speed_slow, 'Fast isn\'t faster than slow!, speed_fast: {}, speed_slow {}'.format(
            speed_fast, speed_slow))
