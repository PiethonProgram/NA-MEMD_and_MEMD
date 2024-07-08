import numpy as np
import pandas as pd


def add_noise(signal, method='na_fix', intensity=0.1, add_rchannel=None):
    # available methods : na_fix, na_snr, na_var, na_ran

    channel_count = signal.shape[0]
    sample_count = signal.shape[1]

    if add_rchannel is None:
        add_rchannel = channel_count

    noise = np.random.randn(add_rchannel, sample_count)

    print(signal.shape,noise.shape)

    if method == 'na_fix':
        pass

    elif method == 'na_snr':
        pass
    elif method == 'na_var':
        pass
    elif method == 'na_ran':
        pass
    else:
        raise ValueError('Invalid method. \n Available methods: na_fix, na_snr, na_var, na_ran')


def na_memd(signals):

    channel_count = signals.shape[0]
    sample_count = signals.shape[1]
    for i in range(channel_count):

        signals[i] = add_noise(signals)