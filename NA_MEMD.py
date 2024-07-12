from noise import *
# from calculations import *
from calculations_orig import *

def na_memd(signal, n_dir=50, stop_crit='stop', stop_vect=(0.075, 0.75, 0.075), n_iter=2, n_imf=None,
            method='memd', intensity=0.1, add_rchannel=None):

    new_signals = add_noise(signal, method=method, intensity=intensity, add_rchannel=add_rchannel)
    imfs = memd(new_signals, n_dir, stop_crit, stop_vect, n_iter, n_imf)

    return imfs


def memd(signal, n_dir=50, stop_crit='stop', stop_vec=(0.075, 0.75, 0.075), n_iter=2, n_imf=100):

    # if np.shape(signal)[0] < np.shape(signal)[1]:
    #     signal = signal.T

    seq, t, nbit, MAXITERATIONS, N_dim, N = initialize_parameters(signal, n_dir)
    sd, sd2, tol = stop_vec

    q = []


    while not stop_emd(signal, seq, n_dir, N_dim):
        print("working")
        m = signal

        # Compute mean and stopping criterion
        counter = 0
        if stop_crit == 'stop':
            stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
        else:
            stop_sift, env_mean, counter = fix(m, t, seq, n_dir, n_iter, counter, N, N_dim)

        # Check for too small amplitude
        if np.max(np.abs(m)) < 1e-10 * np.max(np.abs(signal)):
            print('Forced stop of EMD: too small amplitude')
            break

        # Sifting loop
        while not stop_sift and nbit < MAXITERATIONS:
            print("iter")
            # Sifting
            m = m - env_mean.T  # Transpose env_mean to match the shape of m

            # Compute mean and stopping criterion
            if stop_crit == 'stop':
                stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
            else:
                stop_sift, env_mean, counter = fix(m, t, seq, n_dir, n_iter, counter, N, N_dim)

            nbit += 1

            if nbit == (MAXITERATIONS - 1) and nbit > 100:
                print(nbit, MAXITERATIONS)
                warnings.warn('Forced stop of sifting: too many iterations', UserWarning)

        q.append(m.T)
        signal = signal - m
        nbit = 0

    # Store the residue
    q.append(signal.T)
    q = np.asarray(q)
    return q


def initialize_parameters(signal, n_dir):
    N_dim = signal.shape[0]  # Number of channels
    N = signal.shape[1]  # Number of data points

    base = np.full(N_dim, -n_dir, dtype=np.int64)
    if N_dim == 3:
        base[1] = 2
    else:
        prm = nth_prime(N_dim - 1)
        base[1:] = prm[:N_dim-1]

    seq = np.zeros((n_dir, N_dim))
    for i in range(N_dim):
        seq[:, i] = hamm(n_dir, base[i])

    t = np.arange(1, N + 1)
    nbit = 0
    MAXITERATIONS = 20000  # Increased the maximum number of iterations

    return seq, t, nbit, MAXITERATIONS, N_dim, N



if __name__ == "__main__":
    pass