from unittest import TestCase
import numpy as np

import audio_dspy as adsp

_tolerance_ = 0.001
_N_ = 512


class TestNLs(TestCase):
    def run_samples(self, func, gain, exp):
        high = func(gain)
        low = func(-gain)
        self.assertTrue(np.abs(high - exp) < _tolerance_,
                        'Expected: {}. Actual: {}'.format(exp, high))
        self.assertTrue(np.abs(low + exp) < _tolerance_,
                        'Expected: -{}. Actual: {}'.format(exp, low))

    def run_block(self, func, gain, exp):
        block = np.zeros(_N_)
        block[:int(_N_/2)] = gain
        block[int(_N_/2):] = -gain
        out = func(block)
        self.assertTrue(np.abs(out[0] - exp) < _tolerance_,
                        'Expected: {}. Actual: {}'.format(exp, out[0]))
        self.assertTrue(np.abs(out[_N_-1] + exp) < _tolerance_,
                        'Expected: -{}. Actual: {}'.format(exp, out[_N_-1]))

    def test_soft_clipper(self):
        def func(x): return adsp.soft_clipper(x, 5)
        self.run_samples(func, 10000, 0.8)

    def test_soft_clipper_block(self):
        def func(x): return adsp.soft_clipper(x, 5)
        self.run_block(func, 10000, 0.8)

    def test_hard_clipper(self):
        def func(x): return adsp.hard_clipper(x)
        self.run_samples(func, 10000, 1.0)

    def test_hard_clipper_block(self):
        def func(x): return adsp.hard_clipper(x)
        self.run_block(func, 10000, 1.0)
