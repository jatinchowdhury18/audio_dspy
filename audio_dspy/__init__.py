import numpy as np
from .transfer_function_tools import *
from .prony import *
from .sweeps import *
from .nonlinearities import *
from .plotting import *
from .eq_design import *
name = "audio_dspy"


def normalize(x):
    return x / np.max(np.abs(x))
