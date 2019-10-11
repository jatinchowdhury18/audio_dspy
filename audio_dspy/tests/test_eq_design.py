from unittest import TestCase
import numpy as np

import audio_dspy as adspy

_tolerance_ = 0.001
_fs_ = 44100

class TestEQ(TestCase):
    def test_eq_LPF2(self):
        b, a = adspy.design_LPF2 (_fs_/2, 0.707, _fs_)
        self.assertTrue(np.abs (b[0] - 1.0) < _tolerance_, 'Expected: b[0] = {}. Actual: b[0] = {}'.format (1.0, b[0]))
        self.assertTrue(np.abs (b[1] - 2.0) < _tolerance_, 'Expected: b[1] = {}. Actual: b[1] = {}'.format (2.0, b[1]))
        self.assertTrue(np.abs (b[2] - 1.0) < _tolerance_, 'Expected: b[2] = {}. Actual: b[2] = {}'.format (1.0, b[2]))
        self.assertTrue(np.abs (a[0] - 1.0) < _tolerance_, 'Expected: a[0] = {}. Actual: a[0] = {}'.format (1.0, a[0]))
        self.assertTrue(np.abs (a[1] - 2.0) < _tolerance_, 'Expected: a[1] = {}. Actual: a[1] = {}'.format (2.0, a[1]))
        self.assertTrue(np.abs (a[2] - 1.0) < _tolerance_, 'Expected: a[2] = {}. Actual: a[2] = {}'.format (1.0, a[2]))

    def test_eq_HPF2(self):
        b, a = adspy.design_HPF2 (1, 0.707, _fs_)
        self.assertTrue(np.abs (b[0] - 1.0)  < _tolerance_, 'Expected: b[0] = {}. Actual: b[0] = {}'.format (1.0, b[0]))
        self.assertTrue(np.abs (b[1] - -2.0) < _tolerance_, 'Expected: b[1] = {}. Actual: b[1] = {}'.format (-2.0, b[1]))
        self.assertTrue(np.abs (b[2] - 1.0)  < _tolerance_, 'Expected: b[2] = {}. Actual: b[2] = {}'.format (1.0, b[2]))
        self.assertTrue(np.abs (a[0] - 1.0)  < _tolerance_, 'Expected: a[0] = {}. Actual: a[0] = {}'.format (1.0, a[0]))
        self.assertTrue(np.abs (a[1] - -2.0) < _tolerance_, 'Expected: a[1] = {}. Actual: a[1] = {}'.format (-2.0, a[1]))
        self.assertTrue(np.abs (a[2] - 1.0)  < _tolerance_, 'Expected: a[2] = {}. Actual: a[2] = {}'.format (1.0, a[2]))

    def test_eq_lowshelf(self):
        b, a = adspy.design_lowshelf (_fs_/2, 0.707, 1, _fs_)
        self.assertTrue(np.abs (b[0] - 1.0) < _tolerance_, 'Expected: b[0] = {}. Actual: b[0] = {}'.format (1.0, b[0]))
        self.assertTrue(np.abs (b[1] - 2.0) < _tolerance_, 'Expected: b[1] = {}. Actual: b[1] = {}'.format (2.0, b[1]))
        self.assertTrue(np.abs (b[2] - 1.0) < _tolerance_, 'Expected: b[2] = {}. Actual: b[2] = {}'.format (1.0, b[2]))
        self.assertTrue(np.abs (a[0] - 1.0) < _tolerance_, 'Expected: a[0] = {}. Actual: a[0] = {}'.format (1.0, a[0]))
        self.assertTrue(np.abs (a[1] - 2.0) < _tolerance_, 'Expected: a[1] = {}. Actual: a[1] = {}'.format (2.0, a[1]))
        self.assertTrue(np.abs (a[2] - 1.0) < _tolerance_, 'Expected: a[2] = {}. Actual: a[2] = {}'.format (1.0, a[2]))

    def test_eq_highshelf(self):
        b, a = adspy.design_highshelf (1, 0.707, 1, _fs_)
        self.assertTrue(np.abs (b[0] - 1.0) < _tolerance_, 'Expected: b[0] = {}. Actual: b[0] = {}'.format (1.0, b[0]))
        self.assertTrue(np.abs (b[1] - -2.0) < _tolerance_, 'Expected: b[1] = {}. Actual: b[1] = {}'.format (-2.0, b[1]))
        self.assertTrue(np.abs (b[2] - 1.0) < _tolerance_, 'Expected: b[2] = {}. Actual: b[2] = {}'.format (1.0, b[2]))
        self.assertTrue(np.abs (a[0] - 1.0) < _tolerance_, 'Expected: a[0] = {}. Actual: a[0] = {}'.format (1.0, a[0]))
        self.assertTrue(np.abs (a[1] - -2.0) < _tolerance_, 'Expected: a[1] = {}. Actual: a[1] = {}'.format (-2.0, a[1]))
        self.assertTrue(np.abs (a[2] - 1.0) < _tolerance_, 'Expected: a[2] = {}. Actual: a[2] = {}'.format (1.0, a[2]))

    def test_eq_notch(self):
        b, a = adspy.design_notch (_fs_/2, 0.707, _fs_)
        self.assertTrue(np.abs (b[0] - 1.0) < _tolerance_, 'Expected: b[0] = {}. Actual: b[0] = {}'.format (1.0, b[0]))
        self.assertTrue(np.abs (b[1] - 2.0) < _tolerance_, 'Expected: b[1] = {}. Actual: b[1] = {}'.format (2.0, b[1]))
        self.assertTrue(np.abs (b[2] - 1.0) < _tolerance_, 'Expected: b[2] = {}. Actual: b[2] = {}'.format (1.0, b[2]))
        self.assertTrue(np.abs (a[0] - 1.0) < _tolerance_, 'Expected: a[0] = {}. Actual: a[0] = {}'.format (1.0, a[0]))
        self.assertTrue(np.abs (a[1] - 2.0) < _tolerance_, 'Expected: a[1] = {}. Actual: a[1] = {}'.format (2.0, a[1]))
        self.assertTrue(np.abs (a[2] - 1.0) < _tolerance_, 'Expected: a[2] = {}. Actual: a[2] = {}'.format (1.0, a[2]))

    def test_eq_bell(self):
        b, a = adspy.design_bell (_fs_/2, 0.707, 1, _fs_)
        self.assertTrue(np.abs (b[0] - 1.0) < _tolerance_, 'Expected: b[0] = {}. Actual: b[0] = {}'.format (1.0, b[0]))
        self.assertTrue(np.abs (b[1] - 2.0) < _tolerance_, 'Expected: b[1] = {}. Actual: b[1] = {}'.format (2.0, b[1]))
        self.assertTrue(np.abs (b[2] - 1.0) < _tolerance_, 'Expected: b[2] = {}. Actual: b[2] = {}'.format (1.0, b[2]))
        self.assertTrue(np.abs (a[0] - 1.0) < _tolerance_, 'Expected: a[0] = {}. Actual: a[0] = {}'.format (1.0, a[0]))
        self.assertTrue(np.abs (a[1] - 2.0) < _tolerance_, 'Expected: a[1] = {}. Actual: a[1] = {}'.format (2.0, a[1]))
        self.assertTrue(np.abs (a[2] - 1.0) < _tolerance_, 'Expected: a[2] = {}. Actual: a[2] = {}'.format (1.0, a[2]))
