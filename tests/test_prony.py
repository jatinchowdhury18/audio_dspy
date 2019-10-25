from unittest import TestCase
import numpy as np
import scipy.signal as signal
import random as r

import audio_dspy as adsp

_N_ = 1024
_fs_ = 44100
_rho_ = 0.5
_tolerance_ = 0.0001


class TestProny(TestCase):
    def setUp(self):
        self.h = np.zeros(_N_)
        r.seed(0x3456)
        for n in range(_N_):
            self.h[n] = r.random() - 0.5

    def test_prony(self):
        imp = adsp.impulse(_N_)
        b, a = adsp.design_lowshelf(10000, 2, 2, _fs_)
        x = signal.lfilter(b, a, imp)

        bp, ap = adsp.prony(x, 2, 2)
        _, H = signal.freqz(b, a, fs=_fs_)
        _, Hp = signal.freqz(bp, ap, fs=_fs_)

        H = adsp.normalize(H)
        Hp = adsp.normalize(Hp)

        diffs = 0
        for n, _ in enumerate(H):
            diffs += np.abs(np.abs(H[n]) - np.abs(Hp[n]))
        error = diffs / len(H)
        self.assertTrue(error < _tolerance_, 'Error: {}'.format(error))

    def test_prony_warped(self):
        imp = adsp.impulse(_N_)
        b, a = adsp.design_lowshelf(100, 2, 2, _fs_)
        x = signal.lfilter(b, a, imp)

        bp, ap = adsp.prony_warped(x, 2, 2, _rho_)
        _, H = signal.freqz(b, a, fs=_fs_)
        _, Hp = signal.freqz(bp, ap, fs=_fs_)

        H = adsp.normalize(H)
        Hp = adsp.normalize(Hp)

        diffs = 0
        for n, _ in enumerate(H):
            diffs += np.abs(np.abs(H[n]) - np.abs(Hp[n]))
        error = diffs / len(H)
        # @TODO: reduce this error!
        self.assertTrue(error < _tolerance_*10000, 'Error: {}'.format(error))

    def test_warp_roots(self):
        b_s = np.array([0, 0, 0.00111784])
        a_s = np.array([3.015e-7, 4.721e-4, 4e3])

        b, a = adsp.bilinear_biquad(b_s, a_s, _fs_, matchPole=True)
        pole_freq_uw = np.angle(np.roots(a))[0]

        a_w = adsp.allpass_warp_roots(_rho_, a)
        pole_freq_w = np.angle(np.roots(a_w))[0]

        test_angle = np.arctan2((1 - _rho_**2) * np.sin(pole_freq_uw),
                                (1 + _rho_**2) * np.cos(pole_freq_uw) - 2 * _rho_)

        self.assertTrue(np.abs(pole_freq_w - test_angle) < _tolerance_,
                        'Expected: {}, Actual: {}'.format(test_angle, pole_freq_w))

#    def test_warp(self):
#        test_zero = np.angle (np.roots (self.h))[0]
#        test_angle = np.arctan2 ((1 - _rho_**2) * np.sin (test_zero), (1 + _rho_**2) * np.cos (test_zero) - 2 * _rho_)
#
#        htest = adsp.allpass_warp (_rho_, self.h)
#        check_angle = np.angle (np.roots (htest))[0]
#
#        self.assertTrue (np.abs (test_angle - check_angle) < _tolerance_, 'Expected: {}, Actual: {}'.format (test_angle, check_angle))
