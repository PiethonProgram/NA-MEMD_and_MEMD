from NA_MEMD import *
import time

signal = np.random.randn(10, 1000)
st = time.time()
imfs = memd(signal)
en = time.time()
net = en-st
print("time taken : ", net)
print(imfs.shape)


# Future Updates
#   - Energy Criterion, Residual Criterion
#   - linear interpolation (interp1d) option
#   - speed-up using numba
#   - general speedup of programs
#   - Alternative EMD options
#   -

