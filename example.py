from NA_MEMD import *
import time


np.random.seed(15)
signal = np.random.randn(3, 1000)
st = time.time()
imfs = na_memd(signal, method="na_fix", stop_crit='stop')
en = time.time()
net = en-st
print("time taken : ", net)
print(imfs.shape)
