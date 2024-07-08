import numpy as np


def add_noise(signal, method='na_fix', intensity=0.1, add_rchannel=None):
    # available methods : na_fix, na_snr, na_var, na_ran

    channel_count = signal.shape[0]
    sample_count = signal.shape[1]

    if add_rchannel is None:
        add_rchannel = channel_count

    noise = np.random.randn(add_rchannel, sample_count)

    if method == 'na_fix':
        fix_noise = noise * intensity
        output = np.vstack((signal, fix_noise))
        return output

    elif method == 'na_snr':
        pass

    elif method == 'na_var':
        var_noise = np.var(a=signal, axis=1, keepdims=True) * intensity
        var_noise = noise * var_noise
        output = np.vstack((signal, var_noise))
        return output

    elif method == 'na_ran':
        pass

    else:
        raise ValueError('\nInvalid method. \n'
                         'Available methods: na_fix, na_snr, na_var, na_ran')

    raise ValueError('Error, Review Input')
