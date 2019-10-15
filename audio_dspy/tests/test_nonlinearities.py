from unittest import TestCase
import numpy as np

import audio_dspy as adsp

_tolerance_ = 0.001

class TestNLs(TestCase):
    def test_soft_clipper(self):
        high = adsp.soft_clipper (10000, 5)
        low = adsp.soft_clipper (-10000, 5)
        self.assertTrue(np.abs (high - 0.8)  < _tolerance_, 'Expected: 1. Actual: {}'.format (high))
        self.assertTrue(np.abs (low + 0.8)  < _tolerance_, 'Expected: -1. Actual: {}'.format (low))

    def test_hard_clipper(self):
        high = adsp.hard_clipper (10000)
        low = adsp.hard_clipper (-10000)
        self.assertTrue(np.abs (high - 1.0)  < _tolerance_, 'Expected: 1. Actual: {}'.format (high))
        self.assertTrue(np.abs (low + 1.0)  < _tolerance_, 'Expected: -1. Actual: {}'.format (low))
