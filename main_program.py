import numpy as np
import modul_feature_extraction as mfe
import modul_datapreparation as pre
import matplotlib.pyplot as plt
from modul_wavelet_denoising import dwt

path = '/home/han/S.Kom/Database/data_qtdb/'

raw_data = pre.load(path, '100_1000s.csv')
r = pre.load(path, '100_1000s_rpeak.csv')
fs = 250
rpeaks = r.astype(int)
r_peak = list(rpeaks)

dataset = dwt(raw_data)
#pre.save(new_signal, path, '100_clean.csv')

new_signal = dataset['sinyal']
D2 = dataset['D2']
D3 = dataset['D3']
D4 = dataset['D4']

d234 = list(D2 + D3 + D4)
D3 = D3[57:]
d234 = d234[57:]

waves = mfe.extract(new_signal, r_peak, d234, D3, fs, plot=True)
