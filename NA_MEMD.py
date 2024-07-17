from noise import *
from calculations_backup import *


def na_memd(signal, n_dir=50, stop_crit='stop', stop_vect=(0.075, 0.75, 0.075), n_iter=2, n_imf=None,
            method='memd', intensity=0.1, add_rchannel=None):
    new_signals = add_noise(signal, method=method, intensity=intensity, add_rchannel=add_rchannel)
    imfs = memd(new_signals, n_dir, stop_crit, stop_vect)
    imfs = imfs.transpose(1, 0, 2)
    return imfs


def memd(signal, n_dir=50, stop_crit='stop', stop_vec=(0.075, 0.75, 0.075), n_iter=2, max_imf=100):

    if len(signal) == 0:
        sys.exit('emptyDataSet. Data set cannot be empty.')

    if signal.shape[0] < signal.shape[1]:
        signal = signal.T
    N_dim = signal.shape[1]

    if N_dim < 3:
        sys.exit('Function only processes the signal having more than 3 channels.')
    N = signal.shape[0]
    if not isinstance(n_dir, int) or n_dir < 6:
        sys.exit('invalid num_dir. num_dir should be an integer greater than or equal to 6.')
    if not isinstance(stop_crit, str) or (stop_crit != 'stop' and stop_crit != 'fix_h'):
        sys.exit('invalid stop_criteria. stop_criteria should be either fix_h or stop')
    if not isinstance(stop_vec, (list, tuple, np.ndarray)) or any(
            x for x in stop_vec if not isinstance(x, (int, float, complex))):
        sys.exit('invalid stop_vector. stop_vector should be a list with three elements e.g. default is [0.75,0.75,0.75]')
    if stop_crit == 'fix_h' and (not isinstance(n_iter, int) or n_iter < 0):
        sys.exit('invalid stop_count. stop_count should be a nonnegative integer.')
    base = [-n_dir]
    if N_dim == 3:
        base.append(2)
        seq = np.zeros((n_dir, N_dim - 1))
        for it in range(N_dim - 1):
            seq[:, it] = hamm(n_dir, base[it])
    else:
        prm = nth_prime(N_dim - 1)
        for itr in range(1, N_dim):
            base.append(prm[itr - 1])
        seq = np.zeros((n_dir, N_dim))
        for it in range(N_dim):
            seq[:, it] = hamm(n_dir, base[it])
    t = np.arange(1, N + 1)
    nbit = 0
    MAXITERATIONS = 1000
    sd, sd2, tol = stop_vec[0], stop_vec[1], stop_vec[2] if stop_crit == 'stop' else (None, None, None)
    stp_cnt = n_iter if stop_crit == 'fix_h' else None
    r = signal
    n_imf = 1
    imfs = []
    while not stop_emd(r, seq, n_dir, N_dim) and n_imf <= max_imf:
        m = r
        if stop_crit == 'stop':
            stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
        else:
            counter = 0
            stop_sift, env_mean, counter = fix(m, t, seq, n_dir, stp_cnt, counter, N, N_dim)
        if np.max(np.abs(m)) < 1e-10 * np.max(np.abs(signal)):
            if not stop_sift:
                warnings.warn('emd:warning', 'forced stop of EMD : too small amplitude')
            else:
                print('forced stop of EMD : too small amplitude')
            break
        while not stop_sift and nbit < MAXITERATIONS:
            m = m - env_mean
            if stop_crit == 'stop':
                stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
            else:
                stop_sift, env_mean, counter = fix(m, t, seq, n_dir, stp_cnt, counter, N, N_dim)
            nbit += 1
            if nbit == (MAXITERATIONS - 1) and nbit > 100:
                warnings.warn('emd:warning', 'forced stop of sifting : too many iterations')
        imfs.append(m.T)
        n_imf += 1
        r = r - m
        nbit = 0
    imfs.append(r.T)
    return np.asarray(imfs)




if __name__ == "__main__":
    pass
