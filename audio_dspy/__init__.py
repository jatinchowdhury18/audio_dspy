name = "audio_dspy"

from .eq_design import *
from .plotting import *
from .nonlinearities import *
from .sweeps import *
from .prony import *
from .transfer_function_tools import *

import numpy as np
def normalize (x):
    return x / np.max (np.abs (x))
