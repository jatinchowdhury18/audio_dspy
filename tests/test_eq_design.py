from unittest import TestCase
import numpy as np

import audio_dspy as adsp

_tolerance_ = 0.001
_fs_ = 44100

def checkCoefs(tester, b, a, b_exp, a_exp):
    tester.assertTrue(np.abs (b[0] - b_exp[0]) < _tolerance_, 'Expected: b[0] = {}. Actual: b[0] = {}'.format (b_exp[0], b[0]))
    tester.assertTrue(np.abs (b[1] - b_exp[1]) < _tolerance_, 'Expected: b[1] = {}. Actual: b[1] = {}'.format (b_exp[1], b[1]))
    tester.assertTrue(np.abs (b[2] - b_exp[2]) < _tolerance_, 'Expected: b[2] = {}. Actual: b[2] = {}'.format (b_exp[2], b[2]))
    tester.assertTrue(np.abs (a[0] - a_exp[0]) < _tolerance_, 'Expected: a[0] = {}. Actual: a[0] = {}'.format (b_exp[0], a[0]))
    tester.assertTrue(np.abs (a[1] - a_exp[1]) < _tolerance_, 'Expected: a[1] = {}. Actual: a[1] = {}'.format (b_exp[1], a[1]))
    tester.assertTrue(np.abs (a[2] - a_exp[2]) < _tolerance_, 'Expected: a[2] = {}. Actual: a[2] = {}'.format (b_exp[2], a[2]))

class TestEQ(TestCase):
    def test_eq_LPF2(self):
        b, a = adsp.design_LPF2 (_fs_/2, 0.707, _fs_)
        checkCoefs (self, b, a, [1, 2, 1], [1, 2, 1])

    def test_eq_HPF2(self):
        b, a = adsp.design_HPF2 (1, 0.707, _fs_)
        checkCoefs (self, b, a, [1, -2, 1], [1, -2, 1])

    def test_eq_lowshelf(self):
        b, a = adsp.design_lowshelf (_fs_/2, 0.707, 1, _fs_)
        checkCoefs (self, b, a, [1, 2, 1], [1, 2, 1])

    def test_eq_highshelf(self):
        b, a = adsp.design_highshelf (1, 0.707, 1, _fs_)
        checkCoefs (self, b, a, [1, -2, 1], [1, -2, 1])

    def test_eq_notch(self):
        b, a = adsp.design_notch (_fs_/2, 0.707, _fs_)
        checkCoefs (self, b, a, [1, 2, 1], [1, 2, 1])

    def test_eq_bell(self):
        b, a = adsp.design_bell (_fs_/2, 0.707, 1, _fs_)
        checkCoefs (self, b, a, [1, 2, 1], [1, 2, 1])

    def test_bilinear_biquad(self):
        b_s = np.array ([0, 0, 0.00111784])
        a_s = np.array ([3.015e-7, 4.721e-4, 4e3])

        poleFreq = np.imag (np.roots (a_s))[0] / (2*np.pi)

        b, a = adsp.bilinear_biquad (b_s, a_s, _fs_, matchPole=True)
        dpoleFreq = np.angle (np.roots (a))[0] / (2 * np.pi) * _fs_

        self.assertTrue (np.abs (poleFreq - dpoleFreq) < poleFreq*_tolerance_, 'Pole not matched correctly! Expected: {}, Actual: {}'.format (poleFreq, dpoleFreq))
