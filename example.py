from NA_MEMD import *
import mne



signal = np.random.randn(5,525)
imfs = na_memd(signal)
print(imfs.shape)
