from noise import add_noise


def na_memd(signals,n_proj=50,stop_crit='stop', stop_vect=(0.075, 0.75, 0.075), n_iter=2,
            method='memd', intensity=0.1, add_rchannel=None):

    new_signal = add_noise(signals, method, intensity, add_rchannel)


def input_check():
    pass


def memd():
    pass


if __name__ == "__main__":
    pass