from noise import add_noise
from calculations import *
from visualization import *


def validate_namemd_input(na_method, intensity, add_rchannel):

    if not isinstance(na_method, str):
        sys.exit("Invalid method")

    if not isinstance(intensity, (int, float)) or intensity <= 0:
        sys.exit("Intensity must be a positive number.")

    if add_rchannel is None:
        pass
    else:
        if not isinstance(add_rchannel, int) or add_rchannel <= 0:
            sys.exit("add_rchannel must be a positive integer")

    return True


def validate_memd_input(signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh):

    if len(signal) == 0:
        sys.exit('empty dataset. No Data found')

    if signal.shape[0] < signal.shape[1]:
        signal = signal.T

    N_dim = signal.shape[1]

    if N_dim < 3:
        sys.exit('Function only processes the signal having more than 3 channels.')
    N = signal.shape[0]

    if not isinstance(n_dir, int) or n_dir < 6:
        sys.exit('invalid num_dir. num_dir should be an integer greater than or equal to 6.')
    if not isinstance(stop_crit, str) or (stop_crit not in ['stop', 'fix_h', 'e_diff']):
        sys.exit('invalid stop_criteria. stop_criteria should be either fix_h, stop or e_diff')
    if not isinstance(stop_vec, (list, tuple, np.ndarray)) or any(
            x for x in stop_vec if not isinstance(x, (int, float, complex))):
        sys.exit(
            'invalid stop_vector. stop_vector should be a list with three elements e.g. default is [0.75,0.75,0.75]')
    if stop_crit == 'fix_h' and (not isinstance(n_iter, int) or n_iter < 0):
        sys.exit('invalid stop_count. stop_count should be a nonnegative integer.')

    if not isinstance(max_imf, int) or max_imf < 1:
        sys.exit('invalid max_imf. max_imf should be a positive integer')

    return signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh, N_dim, N


def na_memd(signal, n_dir=50, stop_crit='stop', stop_vect=(0.075, 0.75, 0.075), n_iter=2, max_imf=100, e_thresh=1e-3,
            na_method='na_fix', intensity=0.1, add_rchannel=None, output_condition=False):

    if not validate_namemd_input(na_method, intensity, add_rchannel) :
        sys.exit("Error in Input")
    new_signals = add_noise(signal, na_method=na_method, intensity=intensity, add_rchannel=add_rchannel)
    imfs = memd(new_signals, n_dir, stop_crit, stop_vect, n_iter, max_imf, e_thresh)
    if not output_condition:    # display how many imfs
        imfs = imfs[signal.shape[0]::]

    return imfs


def memd(signal, n_dir=50, stop_crit='stop', stop_vec=(0.05, 0.5, 0.05), n_iter=3, max_imf=100, e_thresh=1e-3):

    signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh, N_dim,  N = (
        validate_memd_input(signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh))

    base = [-n_dir]

    if N_dim == 3:
        base = np.array([-n_dir, 2])
        seq = np.zeros((n_dir, N_dim - 1))
        seq[:, 0] = hamm(n_dir, base[0])
        seq[:, 1] = hamm(n_dir, base[1])
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
    e_thresh = e_thresh if stop_crit == 'e_diff' else None
    r = signal
    n_imf = 1
    imfs = []
    prev_imf = None

    while not stop_emd(r, seq, n_dir, N_dim) and n_imf <= max_imf:
        m = r.copy()
        if stop_crit == 'stop':
            stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
        elif stop_crit == 'fix_h':
            counter = 0
            stop_sift, env_mean, counter = fix(m, t, seq, n_dir, stp_cnt, counter, N, N_dim)
        elif stop_crit == 'e_diff':
            # If there is a previous IMF, calculate the energy difference
            if prev_imf is not None:
                stop_sift, env_mean = e_diff(prev_imf, m, t, seq, n_dir, N, N_dim, e_thresh)
            else:
                # For the first IMF, use a regular stopping criterion
                stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)

        if np.max(np.abs(m)) < 1e-10 * np.max(np.abs(signal)):
            if not stop_sift:
                print('emd:warning : forced stop of EMD : amplitude too small')
            else:
                print('forced stop of EMD : amplitude too small')
            break

        while not stop_sift and nbit < MAXITERATIONS:
            m -= env_mean
            if stop_crit == 'stop':
                stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
            elif stop_crit == 'fix_h':
                stop_sift, env_mean, counter = fix(m, t, seq, n_dir, stp_cnt, counter, N, N_dim)
            elif stop_crit == 'e_diff':
                if prev_imf is not None:
                    stop_sift, env_mean = e_diff(prev_imf, m, t, seq, n_dir, N, N_dim, e_thresh)
                else:
                    stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
            nbit += 1
            if nbit == (MAXITERATIONS - 1) and nbit > 100:
                print('emd:warning : forced stop of sifting : too many iterations')

        imfs.append(m.T)
        n_imf += 1
        r = r - m
        nbit = 0
        # prev_imf = m  # Update prev_imf after extracting the IMF

    imfs.append(r.T)

    return np.asarray(imfs).transpose(1, 0, 2)

