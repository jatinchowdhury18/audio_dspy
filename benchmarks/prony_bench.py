import audio_dspy as adsp
import numpy as np
import random as r

_N_ = 1024
_num_ = 100
_deg_ = 6
_rho_ = 0.5

class PronyTimeSuite:
    """
    Benchmarking Suite for Prony functions
    """
    def setup(self):
        self.h = np.zeros (_N_)
        r.seed (0x3456)
        for n in range (_N_):
            self.h[n] = r.random() - 0.5
        self.b = [r.random(), r.random(), r.random()]

    def time_prony(self):
        for _ in range(_num_):
            adsp.prony (self.h, _deg_, _deg_)

    def time_warp(self):
        for _ in range(_num_):
            adsp.allpass_warp (_rho_, self.h)

    def time_warp_roots(self):
        for _ in range(_num_):
            adsp.allpass_warp_roots (_rho_, self.b)
    
    def time_prony_warp(self):
        for _ in range(_num_):
            adsp.prony_warped (self.h, _deg_, _deg_, _rho_)
