import audio_dspy as adsp
import numpy as np
import random as r

_N_ = 2048
_num_ = 1000


class TFTimeSuite:
    """
    Benchmarking Suite for tranfer function utility functions
    """

    def setup(self):
        self.h = np.zeros(_N_)
        r.seed(0x2345)
        for n in range(_N_):
            self.h[n] = r.random() - 0.5

    def time_tf2linphase(self):
        for _ in range(_num_):
            adsp.tf2linphase(self.h)

    def time_tf2minphase(self):
        for _ in range(_num_):
            adsp.tf2minphase(self.h)
