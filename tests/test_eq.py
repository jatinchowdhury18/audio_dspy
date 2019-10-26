from unittest import TestCase
import numpy as np
import scipy.signal as signal

import audio_dspy as adsp

_N_ = 1024
_fs_ = 44100
_tolerance_ = 0.001


def check_freq_point(tester, freq, exp_mag, w, H):
    idx = np.argmin(np.abs(H - freq))
    tester.assertTrue(np.abs(np.abs(H[idx]) - exp_mag),
                      'Expected: H = {}, Actual: H = {}'.format(exp_mag, np.abs(H[idx])))


class TestEQ(TestCase):
    def setUp(self):
        self.h = adsp.impulse(_N_)
        self.worN = np.logspace(1, 3.3, num=1000, base=20)

    def test_eq_LPF(self):
        eq = adsp.EQ(_fs_)
        eq.add_LPF(1000, 0.707)
        y = eq.process_block(self.h)

        w, H = signal.freqz(y, [1], worN=self.worN, fs=_fs_)
        check_freq_point(self, 1000, 0.7071, w, H)
        check_freq_point(self, 20, 1, w, H)
        check_freq_point(self, 20000, 0, w, H)

    def test_eq_HPF(self):
        eq = adsp.EQ(_fs_)
        eq.add_HPF(1000, 0.707)
        y = eq.process_block(self.h)

        w, H = signal.freqz(y, [1], worN=self.worN, fs=_fs_)
        check_freq_point(self, 1000, 0.7071, w, H)
        check_freq_point(self, 20, 0, w, H)
        check_freq_point(self, 20000, 1, w, H)

    def test_eq_notch(self):
        eq = adsp.EQ(_fs_)
        eq.add_notch(1000, 0.707)
        y = eq.process_block(self.h)

        w, H = signal.freqz(y, [1], worN=self.worN, fs=_fs_)
        check_freq_point(self, 1000, 0, w, H)
        check_freq_point(self, 20, 1, w, H)
        check_freq_point(self, 20000, 1, w, H)

    def test_eq_bell(self):
        eq = adsp.EQ(_fs_)
        eq.add_bell(1000, 0.707, 2)
        y = eq.process_block(self.h)

        w, H = signal.freqz(y, [1], worN=self.worN, fs=_fs_)
        check_freq_point(self, 1000, 2, w, H)
        check_freq_point(self, 20, 1, w, H)
        check_freq_point(self, 20000, 1, w, H)

    def test_eq_lowshelf(self):
        eq = adsp.EQ(_fs_)
        eq.add_lowshelf(1000, 0.707, 2)
        y = eq.process_block(self.h)

        w, H = signal.freqz(y, [1], worN=self.worN, fs=_fs_)
        check_freq_point(self, 1000, 1, w, H)
        check_freq_point(self, 20, 2, w, H)
        check_freq_point(self, 20000, 1, w, H)

    def test_eq_highshelf(self):
        eq = adsp.EQ(_fs_)
        eq.add_highshelf(1000, 0.707, 2)
        y = eq.process_block(self.h)

        w, H = signal.freqz(y, [1], worN=self.worN, fs=_fs_)
        check_freq_point(self, 1000, 1, w, H)
        check_freq_point(self, 20, 1, w, H)
        check_freq_point(self, 20000, 2, w, H)

    def test_reset(self):
        eq = adsp.EQ(_fs_)
        eq.add_notch(1000, 0.707)
        eq.add_lowshelf(1000, 0.707, 2)
        eq.process_block(self.h)
        eq.reset()

        for f in eq.filters:
            self.assertTrue(f.has_been_reset(),
                            'Not all EQ filters have been reset!')
