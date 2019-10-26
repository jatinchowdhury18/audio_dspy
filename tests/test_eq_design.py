from unittest import TestCase
import numpy as np
import scipy.signal as signal

import audio_dspy as adsp

_tolerance_ = 0.001
_fs_ = 44100


def checkCoefs(tester, b, a, b_exp, a_exp):
    for n, _ in enumerate(b):
        tester.assertTrue(np.abs(b[n] - b_exp[n]) < _tolerance_,
                          'Expected: b[{}] = {}. Actual: b[{}] = {}'.format(n, b_exp[n], n, b[n]))

    for n, _ in enumerate(a):
        tester.assertTrue(np.abs(a[n] - a_exp[n]) < _tolerance_,
                          'Expected: a[{}] = {}. Actual: a[{}] = {}'.format(n, a_exp[n], n, a[n]))


class TestEQDesign(TestCase):
    def test_eq_LPF1_design(self):
        b, a = adsp.design_LPF1(_fs_/2, _fs_)
        checkCoefs(self, b, a, [1, 1], [1, 1])

    def test_eq_LPF2_design(self):
        b, a = adsp.design_LPF2(_fs_/2, 0.707, _fs_)
        checkCoefs(self, b, a, [1, 2, 1], [1, 2, 1])

    def test_eq_LPFN_design(self):
        fs = _fs_*8
        # Check filter slope
        for order in range(1, 20):
            sos = adsp.design_LPFN(1000, 0.7071, order, fs)
            w, H = signal.sosfreqz(sos, fs=fs)
            freq1_ind = np.argmin(np.abs(w - (5000)))
            freq2_ind = np.argmin(np.abs(w - w[freq1_ind]*2))
            exp_diff_dB = -6 * order
            diff_dB = 20 * \
                np.log10(np.abs(H[freq2_ind])) - 20 * \
                np.log10(np.abs(H[freq1_ind]))
            self.assertTrue(np.abs(diff_dB - exp_diff_dB) < _tolerance_*1000,
                            "Incorrect slope! Expected: {}, Actual: {}".format(exp_diff_dB, diff_dB))

    def test_eq_HPF1_design(self):
        b, a = adsp.design_HPF1(1, _fs_)
        checkCoefs(self, b, a, [1, -1], [1, -1])

    def test_eq_HPF2_design(self):
        b, a = adsp.design_HPF2(1, 0.707, _fs_)
        checkCoefs(self, b, a, [1, -2, 1], [1, -2, 1])

    def test_eq_HPFN_design(self):
        fs = _fs_*8
        # Check filter slope
        for order in range(1, 20):
            sos = adsp.design_HPFN(20000, 0.7071, order, fs)
            w, H = signal.sosfreqz(sos, fs=fs)
            freq1_ind = np.argmin(np.abs(w - (15000)))
            freq2_ind = np.argmin(np.abs(w - w[freq1_ind]/2))
            exp_diff_dB = -6 * order
            diff_dB = 20 * \
                np.log10(np.abs(H[freq2_ind])) - 20 * \
                np.log10(np.abs(H[freq1_ind]))
            self.assertTrue(np.abs(diff_dB - exp_diff_dB) < _tolerance_*5000,
                            "Incorrect slope! Expected: {}, Actual: {}".format(exp_diff_dB, diff_dB))

    def test_eq_lowshelf_design(self):
        b, a = adsp.design_lowshelf(_fs_/2, 0.707, 1, _fs_)
        checkCoefs(self, b, a, [1, 2, 1], [1, 2, 1])

    def test_eq_highshelf_design(self):
        b, a = adsp.design_highshelf(1, 0.707, 1, _fs_)
        checkCoefs(self, b, a, [1, -2, 1], [1, -2, 1])

    def test_eq_notch_design(self):
        b, a = adsp.design_notch(_fs_/2, 0.707, _fs_)
        checkCoefs(self, b, a, [1, 2, 1], [1, 2, 1])

    def test_eq_bell_design(self):
        b, a = adsp.design_bell(_fs_/2, 0.707, 1, _fs_)
        checkCoefs(self, b, a, [1, 2, 1], [1, 2, 1])

    def test_bilinear_biquad(self):
        b_s = np.array([0, 0, 0.00111784])
        a_s = np.array([3.015e-7, 4.721e-4, 4e3])

        poleFreq = np.imag(np.roots(a_s))[0] / (2*np.pi)

        b, a = adsp.bilinear_biquad(b_s, a_s, _fs_, matchPole=True)
        dpoleFreq = np.angle(np.roots(a))[0] / (2 * np.pi) * _fs_

        self.assertTrue(np.abs(poleFreq - dpoleFreq) < poleFreq*_tolerance_,
                        'Pole not matched correctly! Expected: {}, Actual: {}'.format(poleFreq, dpoleFreq))
