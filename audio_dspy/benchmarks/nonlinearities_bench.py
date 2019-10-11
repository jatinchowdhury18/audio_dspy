import audio_dspy as adspy
import numpy as np

class NonlinearityTimeSuite:
    """
    Benchmarking Suite for nonlinear functions
    """
    def setup(self):
        self.numbers = np.linspace (-10, 10, num=10000)

    def time_soft_clipper_3(self):
        for n in self.numbers:
            adspy.nonlinearities.soft_clipper (n, 3)
