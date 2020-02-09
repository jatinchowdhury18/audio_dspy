from unittest import TestCase
import numpy as np

import audio_dspy as adsp

_tolerance_ = 0.001
_N_ = 512


class TestNLs(TestCase):
    def run_samples(self, func, gain, exp):
        high = func(gain)
        mid = func(0)
        low = func(-gain)
        self.assertTrue(np.abs(high - exp[0]) < _tolerance_,
                        'Expected: {}. Actual: {}'.format(exp[0], high))
        self.assertTrue(np.abs(mid - exp[1]) < _tolerance_,
                        'Expected: {}. Actual: {}'.format(exp[1], high))
        self.assertTrue(np.abs(low - exp[2]) < _tolerance_,
                        'Expected: -{}. Actual: {}'.format(exp[2], low))

    def run_block(self, func, gain, exp):
        block = np.zeros(_N_)
        block[:_N_//3] = gain
        block[_N_//3:2*_N_//3] = 0
        block[2*_N_//3:] = -gain
        out = func(block)
        self.assertTrue(np.abs(out[0] - exp[0]) < _tolerance_,
                        'Expected: {}. Actual: {}'.format(exp[0], out[0]))
        self.assertTrue(np.abs(out[_N_//2] - exp[1]) < _tolerance_,
                        'Expected: {}. Actual: {}'.format(exp[1], out[_N_//2]))
        self.assertTrue(np.abs(out[_N_-1] - exp[2]) < _tolerance_,
                        'Expected: -{}. Actual: {}'.format(exp[2], out[_N_-1]))

    def test_soft_clipper(self):
        self.run_samples(lambda x: adsp.soft_clipper(
            x, 5), 10000, [0.8, 0, -0.8])

    def test_soft_clipper_block(self):
        self.run_block(lambda x: adsp.soft_clipper(
            x, 5), 10000, [0.8, 0.0, -0.8])

    def test_hard_clipper(self):
        self.run_samples(adsp.hard_clipper, 10000, [1.0, 0, -1.0])

    def test_hard_clipper_block(self):
        self.run_block(adsp.hard_clipper, 10000, [1.0, 0, -1.0])

    def test_dropout(self):
        self.run_samples(adsp.dropout, 0.2, [0.064, 0, -0.064])

    def test_dropout_block(self):
        self.run_block(adsp.dropout, 0.2, [0.064, 0, -0.064])

    def test_hwr(self):
        self.run_samples(adsp.halfWaveRect, 1.0, [1.0, 0.0, 0.0])

    def test_hwr_block(self):
        self.run_block(adsp.halfWaveRect, 1.0, [1.0, 0.0, 0.0])

    def test_dioderect(self):
        self.run_samples(adsp.diodeRect, 1.0, [0.997, 0.0, -0.167])

    def test_dioderect_block(self):
        self.run_block(adsp.diodeRect, 1.0, [0.997, 0.0, -0.167])
