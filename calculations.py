# Hammersley an prime calculations
import numpy as np

def hammersley():
    pass


def nth_prime(nth):     # 46,000 times faster than other packages
    if nth >= 0 :
        if nth <= 5 :
            return (small_primes(nth))
        else :
            return (large_primes(nth))
    else :
        raise ValueError("n must be >= 1 for for list of n prime numbers")


def small_primes(nth):
    list_prime = np.array([2, 3, 5, 7, 11])
    return (list_prime[:nth])


def large_primes(nth):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188

    lim = int(nth * (np.log(nth) + np.log(np.log(nth)))) + 1

    sieve = np.ones(lim // 3 + (lim % 6 == 2), dtype=bool)
    for i in range(1, int(lim ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return (np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)])


def is_prime(num):
    i = 2
    while i**2 <= num:
        if num % i == 0: return False
        i += 1
    return True
