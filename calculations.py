import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import sys
import warnings
from numba import jit


def hamm(n_dir, base):
    seq = np.zeros(n_dir)
    if base > 1:
        seed = np.arange(1, n_dir + 1)
        base_inv = 1 / base
        while np.any(seed != 0):
            digit = np.remainder(seed, base)
            seq += digit * base_inv
            base_inv /= base
            seed = np.floor(seed / base)
    else:
        temp = np.arange(1, n_dir + 1)
        seq = (np.remainder(temp, -base + 1) + 0.5) / -base
    return seq


def nth_prime(nth):  # Calculate and return the first nth primes.
    if nth > 0:
        if nth <= 5:    # call corner cases
            return small_primes(nth)
        else:
            return large_primes(nth)    # call general cases
    else:
        raise ValueError("n must be >= 1 for list of n prime numbers")


def small_primes(nth):      # corner cases for nth prime estimation using log
    list_prime = np.array([2, 3, 5, 7, 11])
    return list_prime[:nth]


def large_primes(nth):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188
    lim = int(nth * (np.log(nth) + np.log(np.log(nth)))) + 1
    sieve = np.ones(lim // 3 + (lim % 6 == 2), dtype=bool)
    for i in range(1, int(lim ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

def peaks(X):
    dX = np.sign(np.diff(X.transpose())).transpose()
    locs_max = np.where(np.logical_and(dX[:-1] > 0, dX[1:] < 0))[0] + 1
    pks_max = X[locs_max]

    return (pks_max, locs_max)


# =============================================================================

def local_peaks(x):
    if all(x < 1e-5):
        x = np.zeros((1, len(x)))

    m = len(x) - 1

    # Calculates the extrema of the projected signal
    # Difference between subsequent elements:
    dy = np.diff(x.transpose()).transpose()
    a = np.where(dy != 0)[0]
    lm = np.where(np.diff(a) != 1)[0] + 1
    d = a[lm] - a[lm - 1]
    a[lm] = a[lm] - np.floor(d / 2)
    a = np.insert(a, len(a), m)
    ya = x[a]

    if len(ya) > 1:
        # Maxima
        pks_max, loc_max = peaks(ya)
        # Minima
        pks_min, loc_min = peaks(-ya)

        if len(pks_min) > 0:
            indmin = a[loc_min]
        else:
            indmin = np.asarray([])

        if len(pks_max) > 0:
            indmax = a[loc_max]
        else:
            indmax = np.asarray([])
    else:
        indmin = np.array([])
        indmax = np.array([])

    return (indmin, indmax)


def stop_emd(r, seq, ndir, N_dim):
    ner = np.zeros(ndir)
    dir_vec = np.zeros(N_dim)

    for it in range(ndir):
        if N_dim != 3:  # Multivariate signal (for N_dim != 3) with Hammersley sequence
            # Linear normalization of Hammersley sequence in the range of -1.0 to 1.0
            b = 2 * seq[it, :] - 1

            # Find angles corresponding to the normalized sequence
            tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[::-1][1:] ** 2))), b[:-1])

            # Find coordinates of unit direction vectors on n-sphere
            dir_vec = np.cumprod(np.concatenate(([1], np.sin(tht))))
            dir_vec[:-1] *= np.cos(tht)
        else:  # Trivariate signal with Hammersley sequence
            # Linear normalization of Hammersley sequence in the range of -1.0 to 1.0
            tt = 2 * seq[it, 0] - 1
            tt = np.clip(tt, -1, 1)

            # Normalize angle from 0 to 2*pi
            phirad = seq[it, 1] * 2 * np.pi
            st = np.sqrt(1.0 - tt * tt)

            dir_vec[0] = st * np.cos(phirad)
            dir_vec[1] = st * np.sin(phirad)
            dir_vec[2] = tt

        # Projection of input signal on nth (out of total ndir) direction vectors
        y = np.dot(r.T, dir_vec)

        # Calculates the extrema of the projected signal
        indmin, indmax = local_peaks(y)

        ner[it] = len(indmin) + len(indmax)

    # Stops if all projected signals have less than 3 extrema
    stp = np.all(ner < 3)
    return stp


def zero_crossings(x):
    indzer = np.where(x[0:-1] * x[1:] < 0)[0]

    if np.any(x == 0):
        iz = np.where(x == 0)[0]
        if np.any(np.diff(iz) == 1):
            zer = x == 0
            dz = np.diff(np.concatenate(([0], zer, [0])))
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2)
        else:
            indz = iz
        indzer = np.sort(np.concatenate((indzer, indz)))

    return indzer


def stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim):
    try:
        env_mean, nem, nzm, amp = envelope_mean(m, t, seq, ndir, N, N_dim)
        sx = np.sqrt(np.sum(np.power(env_mean, 2), axis=1))

        # if np.any(amp):
        #     sx = sx / amp

        if all(amp):  # something is wrong here
            sx = sx / amp

        if np.mean(sx > sd) <= tol and not np.any(sx > sd2) and np.any(nem > 2):
            stp = 1
        else:
            stp = 0
    except Exception as e:
        print(f"Exception in stop function: {e}")
        env_mean = np.zeros((N, N_dim))
        stp = 1

    return stp, env_mean

def fix(m, t, seq, ndir, stp_cnt, counter, N, N_dim):
    try:
        env_mean, nem, nzm, amp = envelope_mean(m, t, seq, ndir, N, N_dim)

        if np.all(np.abs(nzm - nem) > 1):
            stp = 0
            counter = 0
        else:
            counter += 1
            stp = (counter >= stp_cnt)
    except Exception as e:
        print(f"Exception in fix function: {e}")
        env_mean = np.zeros((N, N_dim))
        stp = 1

    return stp, env_mean, counter


def envelope_mean(m, t, seq, ndir, N, N_dim):
    NBSYM = 2
    count = 0

    # Initialize arrays for the mean envelope, amplitude, number of extrema, and number of zero crossings
    env_mean = np.zeros((len(t), N_dim))
    amp = np.zeros((len(t)))
    nem = np.zeros((ndir))
    nzm = np.zeros((ndir))

    dir_vec = np.zeros((N_dim, 1))

    for it in range(0,ndir):
        if N_dim != 3:  # Multivariate signal (for N_dim != 3) with Hammersley sequence
            # Linear normalization of Hammersley sequence in the range of -1.00 to 1.00
            b = 2 * seq[it, :] - 1

            # Find angles corresponding to the normalized sequence
            tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[:0:-1] ** 2))), b[:N_dim - 1]).transpose()

            # Find coordinates of unit direction vectors on n-sphere
            dir_vec[:, 0] = np.cumprod(np.concatenate(([1], np.sin(tht))))
            dir_vec[:N_dim - 1, 0] = np.cos(tht) * dir_vec[:N_dim - 1, 0]
        else:  # Trivariate signal with Hammersley sequence
            # Linear normalization of Hammersley sequence in the range of -1.0 to 1.0
            tt = 2 * seq[it, 0] - 1
            tt = np.clip(tt, -1, 1)

            # Normalize angle from 0 to 2*pi
            phirad = seq[it, 1] * 2 * np.pi
            st = np.sqrt(1.0 - tt * tt)

            dir_vec[0] = st * np.cos(phirad)
            dir_vec[1] = st * np.sin(phirad)
            dir_vec[2] = tt

        # Projection of input signal on nth (out of total ndir) direction vectors
        y = np.dot(m.T, dir_vec).flatten()

        # Calculate the extrema of the projected signal
        indmin, indmax = local_peaks(y)

        nem[it] = len(indmin) + len(indmax)
        indzer = zero_crossings(y)
        nzm[it] = len(indzer)

        tmin, tmax, zmin, zmax, mode = boundary_conditions(indmin, indmax, t, y, m.T, NBSYM)

        # Calculate multidimensional envelopes using spline interpolation
        # Only done if number of extrema of the projected signal exceeds 3
        if mode:
            fmin = CubicSpline(tmin, zmin, bc_type='not-a-knot')
            env_min = fmin(t)
            fmax = CubicSpline(tmax, zmax, bc_type='not-a-knot')
            env_max = fmax(t)
            amp += np.sqrt(np.sum(np.power(env_max - env_min, 2), axis=1)) / 2
            env_mean += (env_max + env_min) / 2
        else:  # If the projected signal has inadequate extrema
            count += 1

    if ndir > count:
        env_mean /= (ndir - count)
        amp /= (ndir - count)
    else:
        env_mean = np.zeros((N, N_dim))
        amp = np.zeros((N))
        nem = np.zeros((ndir))

    return env_mean, nem, nzm, amp


def boundary_conditions(indmin, indmax, t, x, z, nbsym):
    lx = len(x) - 1
    end_max = len(indmax) - 1
    end_min = len(indmin) - 1
    indmin = indmin.astype(int)
    indmax = indmax.astype(int)

    if len(indmin) + len(indmax) < 3:
        mode = 0
        tmin = tmax = zmin = zmax = None
        return (tmin, tmax, zmin, zmax, mode)
    else:
        mode = 1  # the projected signal has inadequate extrema
    # boundary conditions for interpolations :
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = np.flipud(indmax[1:min(end_max + 1, nbsym + 1)])
            lmin = np.flipud(indmin[:min(end_min + 1, nbsym)])
            lsym = indmax[0]

        else:
            lmax = np.flipud(indmax[:min(end_max + 1, nbsym)])
            lmin = np.concatenate((np.flipud(indmin[:min(end_min + 1, nbsym - 1)]), ([0])))
            lsym = 0

    else:
        if x[0] < x[indmax[0]]:
            lmax = np.flipud(indmax[:min(end_max + 1, nbsym)])
            lmin = np.flipud(indmin[1:min(end_min + 1, nbsym + 1)])
            lsym = indmin[0]

        else:
            lmax = np.concatenate((np.flipud(indmax[:min(end_max + 1, nbsym - 1)]), ([0])))
            lmin = np.flipud(indmin[:min(end_min + 1, nbsym)])
            lsym = 0

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0):])
            rmin = np.flipud(indmin[max(end_min - nbsym, 0):-1])
            rsym = indmin[-1]

        else:
            rmax = np.concatenate((np.array([lx]), np.flipud(indmax[max(end_max - nbsym + 2, 0):])))
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0):])
            rsym = lx

    else:
        if x[-1] > x[indmin[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym, 0):-1])
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0):])
            rsym = indmax[-1]

        else:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0):])
            rmin = np.concatenate((np.array([lx]), np.flipud(indmin[max(end_min - nbsym + 2, 0):])))
            rsym = lx

    tlmin = 2 * t[lsym] - t[lmin]
    tlmax = 2 * t[lsym] - t[lmax]
    trmin = 2 * t[rsym] - t[rmin]
    trmax = 2 * t[rsym] - t[rmax]

    # in case symmetrized parts do not extend enough
    if tlmin[0] > t[0] or tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[:min(end_max + 1, nbsym)])
        else:
            lmin = np.flipud(indmin[:min(end_min + 1, nbsym)])
        if lsym == 1:
            sys.exit('bug')
        lsym = 0
        tlmin = 2 * t[lsym] - t[lmin]
        tlmax = 2 * t[lsym] - t[lmax]

    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0):])
        else:
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0):])
        if rsym == lx:
            sys.exit('bug')
        rsym = lx
        trmin = 2 * t[rsym] - t[rmin]
        trmax = 2 * t[rsym] - t[rmax]

    zlmax = z[lmax, :]
    zlmin = z[lmin, :]
    zrmax = z[rmax, :]
    zrmin = z[rmin, :]

    tmin = np.hstack((tlmin, t[indmin], trmin))
    tmax = np.hstack((tlmax, t[indmax], trmax))
    zmin = np.vstack((zlmin, z[indmin, :], zrmin))
    zmax = np.vstack((zlmax, z[indmax, :], zrmax))

    return (tmin, tmax, zmin, zmax, mode)