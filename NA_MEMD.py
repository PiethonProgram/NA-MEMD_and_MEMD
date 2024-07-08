import numpy as np
import pandas as pd
from noise import *


def na_memd(signals):

    channel_count = signals.shape[0]
    sample_count = signals.shape[1]
    for i in range(channel_count):

        signals[i] = add_noise(signals)