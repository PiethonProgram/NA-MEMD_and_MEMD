from NA_MEMD import *

signal = np.random.randn(10, 1600)
imfs = na_memd(signal=signal)
print(imfs.shape)




