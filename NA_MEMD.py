from noise import *


def na_memd(signal,n_dir=50,stop_crit='stop', stop_vect=(0.075, 0.75, 0.075), n_iter=2, n_imf=None,
            method='memd', intensity=0.1, add_rchannel=None):

    new_signals = add_noise(signal,method=method, intensity=intensity, add_rchannel=add_rchannel)
    imfs = memd(new_signals, n_dir, stop_crit, stop_vect, n_iter, n_imf)

    return (imfs)


def memd(*args):
    x, n_dir, stop_crit, stop_vec, n_iter, = input_check(args)
    seq, t,
    signal = x
    return 1


def input_check(*args):
    pass


if __name__ == "__main__":
    pass