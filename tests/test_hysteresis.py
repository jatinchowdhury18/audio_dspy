from unittest import TestCase
import numpy as np

import audio_dspy as adsp

_FS_ = 44100.0
_N_ = 1024


class TestHysteresis(TestCase):
    def test_hysteresis(self):
        freq = 100
        n = np.arange(_N_)
        x = np.sin(2 * np.pi * n * freq / _FS_) * (n/_N_)

        hysteresis = adsp.Hysteresis(1.0, 1.0, 1.0, _FS_, mode='RK4')
        y = hysteresis.process_block(x)

        hysteresis2 = adsp.Hysteresis(1.0, 1.0, 1.0, _FS_, mode='RK2')
        y2 = hysteresis2.process_block(x)

        self.assertTrue(np.sum(np.abs(y - y2)) / _N_ < 5.0e-6,
                        'Hysteresis response is incorrect!')
