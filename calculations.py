# Hammersley a prime calculations
import numpy as np


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


def local_peaks(x):
    if np.all(x < 1e-5):
        x = np.zeros(len(x))

    m = len(x) - 1

    # Calculates the extrema of the projected signal
    dy = np.diff(x)
    a = np.where(dy != 0)[0]
    lm = np.where(np.diff(a) != 1)[0] + 1
    d = a[lm] - a[lm - 1]
    a[lm] = a[lm] - np.floor(d / 2)
    a = np.append(a, m)
    ya = x[a]

    if len(ya) > 1:
        # Calculate differences for peak detection
        dya = np.diff(ya)
        sign_change = np.sign(dya)

        # Find local maxima
        loc_max = np.where((sign_change[:-1] > 0) & (sign_change[1:] < 0))[0] + 1
        indmax = a[loc_max] if len(loc_max) > 0 else np.array([])

        # Find local minima
        loc_min = np.where((sign_change[:-1] < 0) & (sign_change[1:] > 0))[0] + 1
        indmin = a[loc_min] if len(loc_min) > 0 else np.array([])
    else:
        indmin = np.array([])
        indmax = np.array([])

    return indmin, indmax


"""
def is_prime(num):
    i = 2
    while i**2 <= num:
        if num % i == 0: return False
        i += 1
    return True
"""
