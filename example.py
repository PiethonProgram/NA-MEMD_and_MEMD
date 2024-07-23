from NA_MEMD import *
import time

# signal = np.random.randn(10, 1000)
# st = time.time()
# imfs = memd(signal)
# en = time.time()
# net = en-st
# print("time taken : ", net)
# print(imfs.shape)

to_input = 'C:/Ethan/UF Official/FAU REU/Research_Material/extract/na_memd_raw/participant_' + \
    '3_Alpha_imfs.npy'
dan = np.load(to_input)
viz(dan, 30000)


#
# outstart = 'C:/Ethan/UF Official/FAU REU/Research_Material/extract/Th_Al_extract/participant_'
# outend = '_theta_filtered.npy'
# npname = outstart + '3' + outend
# datat = np.load(npname)
# st = time.time()
# imf_1 = na_memd(datat[:, :30000])
# viz(imf_1, 30000)
# en = time.time()
# net = en-st
# print("time taken : ", net)
# print(imf_1.shape)

# Future Updates
#   - Energy Criterion, Residual Criterion
#   - linear interpolation (interp1d) option
#   - speed-up using numba
#   - general speedup of programs

