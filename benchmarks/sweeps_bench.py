import audio_dspy as adsp

_fs_ = 44100
_num_ = 1000
_dur_ = 1

class SweepsTimeSuite:
    """
    Benchmarkng Suite for sweep functions
    """
    def setup(self):
        self.sweep = adsp.sweep_log (1, _fs_/2, _dur_, _fs_)
        self.sweep2 = adsp.sweep_log (1, _fs_/2, _dur_, _fs_)

    def time_log_sweep(self):
        for _ in range(_num_):
            adsp.sweep_log (1, _fs_/2, _dur_, _fs_)

    def time_lin_sweep(self):
        for _ in range(_num_):
            adsp.sweep_lin (_dur_, _fs_)
    
    def time_sweep2ir(self):
        for _ in range(_num_):
            adsp.sweep2ir (self.sweep, self.sweep2)
