import numpy as np


# Available methods : na_fix, na_snr, na_var, na_ran
def add_noise(signal, method='na_fix', intensity=0.1, add_rchannel=None):
    """ Add noise to signals
    Parameters :

    - signal (ndarray): signal input consisting of r rows and c columns where each row represents an input channel
                        and each column represents sample data.
    - method (str) : method of noise to apply to signal.
        memd : memd implementation (do nothing)
        na_fix : fixed noise
        na_snr : signal-to-noise ratio
        na_var : variance based noise
        na_ran : random noise
        w_gauss : white guassian noise
    - intensity (float) : intensity of noise (if using na_snr then intensity becomes desired SNR ratio)
    - add_rchannel (int) : number of noise channels to add to signals (default set equal to number of input channels)

    Output :

    returns ndarray consisting of r + add_rchannel channels of length c
    """

    channel_count, sample_count = signal.shape

    if add_rchannel is None:
        add_rchannel = channel_count

    noise = np.random.randn(add_rchannel, sample_count)

    if method == 'memd':
        return signal

    elif method == 'na_fix':
        fix_noise = noise * intensity
        output = np.vstack((signal, fix_noise))
        return output

    elif method == 'na_snr':
        sig_power = np.sum(np.abs(signal) ** 2)/sample_count
        noise_add = noise * np.sqrt(sig_power/(10**(intensity/10)))
        output = np.vstack((signal, noise_add))
        return output

    elif method == 'na_var':
        var_noise = np.var(a=signal, axis=1, keepdims=True)
        var_noise = noise * var_noise * intensity
        output = np.vstack((signal, var_noise))
        return output

    elif method == 'na_ran':
        rand_factor = np.random.rand(1, sample_count)
        ran_noise = rand_factor * intensity * noise
        output = np.vstack((signal, ran_noise))
        return output
    elif method == 'w_gauss':
        wgn_noise = noise * intensity
        output = np.vstack((signal, wgn_noise))
        return output

    else:
        raise ValueError('\nInvalid method. \n'
                         'Available methods: na_fix, na_snr, na_var, na_ran, memd')
