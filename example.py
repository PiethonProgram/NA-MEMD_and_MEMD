from NA_MEMD import *
import time

signal = np.random.randn(10, 1600)
st = time.time()
imfs = na_memd(signal=signal)
en = time.time()
net = en-st
print("time taken : ", net)
print(imfs.shape)




