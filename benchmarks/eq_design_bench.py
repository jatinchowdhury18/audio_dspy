import audio_dspy as adsp
import numpy as np

_num_ = 1000
_fc_ = 1000
_Q_ = 0.7071
_fs_ = 44100
_gain_ = 2
_N_ = 16


class EQTimeSuite:
    """
    Benchmarkng Suite for EQ design functions
    """

    def time_bell_filter(self):
        for _ in range(_num_):
            adsp.design_bell(_fc_, _Q_, _gain_, _fs_)

    def time_lowshelf_filter(self):
        for _ in range(_num_):
            adsp.design_lowshelf(_fc_, _Q_, _gain_, _fs_)

    def time_highshelf_filter(self):
        for _ in range(_num_):
            adsp.design_highshelf(_fc_, _Q_, _gain_, _fs_)

    def time_notch_filter(self):
        for _ in range(_num_):
            adsp.design_notch(_fc_, _Q_, _fs_)

    def time_LPF1_filter(self):
        for _ in range(_num_):
            adsp.design_LPF1(_fc_, _fs_)

    def time_LPF2_filter(self):
        for _ in range(_num_):
            adsp.design_LPF2(_fc_, _Q_, _fs_)

    def time_LPFN_filter(self):
        for _ in range(_num_):
            adsp.design_LPFN(_fc_, _Q_, _N_,  _fs_)

    def time_HPF1_filter(self):
        for _ in range(_num_):
            adsp.design_HPF1(_fc_, _fs_)

    def time_HPF2_filter(self):
        for _ in range(_num_):
            adsp.design_HPF2(_fc_, _Q_, _fs_)

    def time_HPFN_filter(self):
        for _ in range(_num_):
            adsp.design_HPFN(_fc_, _Q_, _N_,  _fs_)

    def time_bilinear_biquad(self):
        for _ in range(_num_):
            adsp.bilinear_biquad(np.array([1, 1, 1]), np.array(
                [1, 0, 0]), _fs_, matchPole=True)
