from NA_MEMD import *
import time


np.random.seed(15)
signal = np.random.rand(5, 1000)
st = time.time()
imfs = na_memd(signal, method='na_fix', stop_crit='stop')
en = time.time()
net = en-st
print("time taken : ", net)
print(imfs.shape)


# Future Updates
#   - Energy Criterion, Residual Criterion
#   - linear interpolation (interp1d) option
#   - speed-up using numba
#   - general speedup of programs

