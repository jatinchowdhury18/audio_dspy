import audio_dspy as adsp
import numpy as np


class NonlinearityTimeSuite:
    """
    Benchmarking Suite for nonlinear functions
    """

    def setup(self):
        self.numbers = np.linspace(-10, 10, num=10000)

    def time_soft_clipper_3(self):
        for n in self.numbers:
            adsp.nonlinearities.soft_clipper(n, 3)

    def time_hard_clipper(self):
        for n in self.numbers:
            adsp.nonlinearities.hard_clipper(n)
