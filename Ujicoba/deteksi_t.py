"""
PERCOBAAN DETEKSI P dan T based on paper DWT
"""

import numpy as np
from modul_feature_extraction import extract
import modul_datapreparation as pre
import matplotlib.pyplot as plt
from modul_wavelet_denoising import dwt
from math import sqrt
import wfdb
import pandas as pd

####################   BATAS FUNGSI   #########################
"""
def p_peak(sinyal, sinyal_d45, rpeak, qrs_on, fs):
    rr_list = []
    for i in range(20):
        rr_list.append(rpeak[i+1] - rpeak[i])
    rr = np.mean(rr_list)
    osw = qrs_on
"""    
def t_peak(sinyal, sinyal_d45, rpeak, qrs_on, qrs_off, fs):
    rr_list = []
    for i in range(20):
        rr_list.append(rpeak[i+1] - rpeak[i])
    rr = np.mean(rr_list)
    iswt_list = []
    #val_d45 = []
    tmax_d45 = []
    batas_kiri = []
    batas_kanan = []
    tmin_d45 = []
    biphasic = []
    #peak = []
    for i in range(len(rpeak)-1):
        # ISWT SECTION
        a = int(qrs_off[i]+(fs*0.1))
        b = int(qrs_on[i]+(0.46*(sqrt(rr*fs))))
        print ("===== BATAS ITERASI %d =====" % i)
        #print ("Panjang ISWT dari index %d - %d, panjang = %d" % (a, b, b-a))
        iswt = np.array(sinyal[a:b])  #ISWT DENGAN DENOISED SIGNAL
        iswt_d45 = np.array(sinyal_d45[a:b])  #ISWT DENGAN SINYAL REKONSTRUKSI D45
        iswt_list.append(iswt_d45)
        #tmax = np.argmax(iswt)+a 
        
        # MENCARI NILAI ABSOLUT DARI T PEAK (LANGKAH NOMER 1)
        abs_tmax = np.argmax(abs(iswt_d45))+a #INDEX NILAI MAX ABSOLUT di d234 signal
        val_abs_tmax = sinyal_d45[abs_tmax]  #NILAI MAX ABSOLUT di d234 signal
        #val_d45.append(val_abs_tmax)

        # THRESHOLDING
        a1 = a-20
        b1 = b+40
        n = sinyal_d45[a1:b1]
        maks = np.argmax(n)+a1
        thr = 0.125*(abs(sinyal_d45[maks]))
        #print ("NILAI THR1 = %0.3f,  TMAX_ASLI= %0.3f, dan nilai TMAX_ABSOLUT = %0.3f" % (thr1, sinyal[asli], max2))
        
        # PERCABANGAN ( LANGKAH NOMER 2 )
        if val_abs_tmax >= 0:
            tpeak = np.argmax(iswt_d45)+(a+1)
            iswt_new = np.array(sinyal[tpeak:b1])
            t_end = np.argmin(iswt_new)+tpeak
            print ("Nilai T Peak pada sinyal D45 adalah = ", val_abs_tmax)
        elif val_abs_tmax < 0:
            tpeak = np.argmax(abs(iswt_d45))+(a+1)
            print ("[TMAX NEGATIF] Nilai T Peak pada sinyal D45 adalah = ", val_abs_tmax)
            # PERBADINGAN DENGAN THRESHOLD ( LANGKAH NOMER 3)
            if abs(sinyal_d45[tpeak]) > thr:
                if sinyal[tpeak] >= 0:
                    print ("NILAI T MAX DI SINYAL DENOISING POSITIF = ", sinyal[tpeak])
                    #iswt_d45_new = np.array(sinyal_d45[tmax45:b1])
                    t_end = np.argmin(iswt_d45)+(a+1)
                elif sinyal[tpeak] < 0:
                    #iswt_d45_new = np.array(sinyal_d45[tmax45:b1])
                    t_end = np.argmax(iswt_d45)+(a+1)
                    print ("NILAI T MAX DI SINYAL DENOISING NEGATIF = ", sinyal[tpeak])
                    # BAZZET FORMULA UNTUK QT CORRECTION
                    qtc = bazzet(qrs_on[i], tpeak, rpeak[i], rpeak[i+1], fs)
                    #PERCABANGAN KETIGA ( LANGKAH NOMER 5 )
                    if qtc > 0.52:
                        biphasic.append(t_end)
                        tpeak = np.argmin(iswt)+a
                        iswt45_new = np.array(sinyal_d45[tpeak:b])
                        t_end = np.argmax(iswt45_new)+tpeak
                        print ("LONG QT INTERVAl = ", qtc)
                    else:
                        tpeak = np.argmax(iswt)+(a+1)
                        iswt_new = np.array(sinyal_d45[tpeak:b1])
                        t_end = np.argmin(iswt_new)+tpeak
            elif abs(sinyal_d45[tpeak]) < thr:
                print ("Nilai Threshold = ", thr)
                print ("[T MAX DIBAWAH THRESHOLD] = ", sinyal_d45[tpeak])
                tpeak= 2
        
        qtc = bazzet(qrs_on[i], tpeak, rpeak[i], rpeak[i+1], fs)        
        if qtc > 0.52:
            biphasic.append(tpeak)
            tpeak = np.argmin(iswt)+a
            iswt45_new = np.array(sinyal_d45[tpeak:b])
            t_end = np.argmax(iswt45_new)+tpeak
            print ("LONG QT INTERVAl = ", qtc)
        else:
            tpeak = np.argmax(iswt)+a
            iswt_d45new = np.array(sinyal_d45[tpeak:b1])
            t_end = np.argmin(iswt_d45new)+tpeak        
        
        tmax_d45.append(tpeak)
        tmin_d45.append(t_end)
        batas_kiri.append(a)
        batas_kanan.append(b)
    hasil = {'index_max45': tmax_d45, 'index_min45': tmin_d45, 'kiri': batas_kiri, 'kanan': batas_kanan, 't': biphasic}
    #hasil = {'index_max45': tmax, 'index_max_denoised': tmax2, 'max_45': max_1, 'max_denoised': max_2}
    return hasil
    
def bazzet(qrs_on, tmin, rpeaks, rpeaks2, fs):
    #qtc = []
    qt  = (tmin - qrs_on)/fs
    rr_int = sqrt((rpeaks2 - rpeaks)/fs)
    #print ("QT = ", qt)
    #print ("akar rr = ", rr_int)
    long_qt = qt/rr_int      
    #qtc.append(long_qt)
    #print ("QT INTERVAL = ", long_qt)
    return (long_qt)
        
#def tpeak(sinyal, sinyal_recon, qrs_on, qrs_off, a, b):
    
####################   BATAS FUNGSI   #########################

path = '/home/han/S.Kom/Database/data_qtdb/'

raw_data = pre.load(path, '100.csv')
r = pre.load(path, '100_rpeak.csv')
fs = 250
rpeaks = list(r.astype(int))

wavelet = dwt(raw_data)
new_signal = wavelet['sinyal']
D2 = wavelet['D2']
D3 = wavelet['D3']
D4 = wavelet['D4']
D5 = wavelet['D5']
D6 = wavelet['D6']

d234 = list(D2 + D3 + D4)
d45 = list(D4+D5)
d56 = list(D5+D6)
#d2345 = list(D2+D3+D4+D5)
#d2345 = d2345[120:]
D3 = D3[57:]
d234 = d234[57:]
d45 = d45[124:]
#d45 = d45[57:]
#d56 = d56[200:]


qrs = extract(new_signal, rpeaks, d234, D3, fs, plot='false')
rpeak = qrs['rpeak']
q_wave = qrs['q_wave']
s_wave = qrs['s_wave']
qrs_on = qrs['qrs_onset']
qrs_off = qrs['qrs_offset']

ano = wfdb.rdann('/home/han/S.Kom/Database/qtdb/sel100','pu0')
ano_dictionary = ano.__dict__
ann_label = ano_dictionary['symbol']
ann = ano_dictionary['sample']
kamus = {'nilai': ann, 'label': ann_label}
df = pd.DataFrame(kamus)
df.to_csv(r'/home/han/S.Kom/Database/dataframe.csv')

figure1 = wfdb.plot_items(raw_data, [ann])

####################   BATAS KODINGAN LAMA   #########################
#p = p_peak(new_signal, d45, rpeaks, qrs_on, fs)
t = t_peak(new_signal, d45, rpeak, qrs_on, qrs_off, fs)

tmax = t['index_max45']
batas_kiri = t['kiri']
batas_kanan = t['kanan']
tmin = t['index_min45']
tmula = t['t']

longqt = []

for i in range(len(rpeaks)-1):
    data_longqt = bazzet(qrs_on[i], tmin[i], rpeaks[i], rpeaks[i+1], fs)
    longqt.append(data_longqt)
 
#data_longqt = bazzet(, tmin[i], rpeaks[i], rpeaks[i+1], fs)   

####################   BATAS KODINGAN BARU   #########################
plt.figure("SINYAL")
plt.plot(new_signal)
plt.plot(d45, linestyle=":")
#plt.plot(D5, linestyle=":")
#plt.plot(new_signal, markevery=tmax, marker="o", color='black')
#plt.plot(d45, linestyle=":", markevery=tmax, marker="o", color='blue')
plt.plot(new_signal, linestyle="none",markevery=tmax, marker="o", color='red')
plt.plot(d45, linestyle="none",markevery=batas_kiri, marker="o", color='black')
plt.plot(d45, linestyle="none",markevery=batas_kanan, marker="o", color='black')
plt.plot(new_signal, linestyle="none",markevery=tmin, marker="o", color='green')


"""
def t_peak(sinyal, sinyal_d45, rpeak, qrs_on, qrs_off, fs):
    rr_list = []
    for i in range(20):
        rr_list.append(rpeak[i+1] - rpeak[i])
    rr = np.mean(rr_list)
    iswt_list = []
    tmax = []
    tmin = []
    for i in range(len(rpeak)-1):
        a = int(qrs_off[i]+(fs*0.1))
        b = int(qrs_on[i]+(0.46*(sqrt(rr*fs))))
        a1 = a-20
        b1 = b+40
        ISWT = np.array(d45[a:b])
        n = np.array(d45[a:b1])
        iswt_list.append(ISWT)
        t_max = np.argmax(ISWT)+a
        tmax.append(t_max)
        t_min = np.argmin(n)+a
        tmin.append(t_min)
    hasil = {'tmin': tmin, 'tmax': tmax}
    return hasil
"""
