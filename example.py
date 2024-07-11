from NA_MEMD import *
import mne
import time


signal = np.random.randn(32, 1600)
st = time.time()
imfs = memd(signal)
en = time.time()
net = en-st
print("time taken : ", net)
print(imfs.shape)


outstart = 'C:/Ethan/UF Official/FAU REU/Research_Material/Article/derivatives/'
outend = '_task-eyesclosed_eeg.set'
filenames = outstart + 'sub-003' + '/eeg/sub-003' + outend

raw_01 = mne.io.read_raw_eeglab(filenames, preload=True)
data_raw_01 = raw_01.get_data()

st = time.time()
imf_1 = na_memd(data_raw_01)
en = time.time()
net = en-st
print("time taken : ", net)
print(imf_1.shape)


