from noise import *


def na_memd(signals):

    channel_count = signals.shape[0]
    sample_count = signals.shape[1]

    noisy_signal1 = add_noise(signals, method='na_fix')
    print("success", noisy_signal1.shape)
    noisy_signal2 = add_noise(signals, method='na_snr')
    print("success", noisy_signal2.shape)
    noisy_signal3 = add_noise(signals, method='na_var')
    print("success", noisy_signal3.shape)
    noisy_signal4 = add_noise(signals, method='na_ran')
    print("success", noisy_signal4.shape)



if __name__ == "__main__":
    pass