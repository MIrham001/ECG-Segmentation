"""
MODUL EKSTRAKSI FITUR TERBARU
Modul untuk deteksi PQRST (Feature Extraction)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def extract(data, rpeak, sinyal_rekonstruksi_d234, sinyal_rekonstruksi_d3, sinyal_d45, fs, plot):
    r_peak = rpeak_correction(data, rpeak, fs)
    q_waves = func_qwave(data, rpeak, fs)
    s_waves = func_swave(data, rpeak, fs)
    t_waves = func_twave(data, rpeak, fs)
    #p_waves = func_pwave(data, t_waves, rpeak, fs)
    
    qrs_set = func_qrs_set(data, rpeak, fs, sinyal_rekonstruksi_d234, sinyal_rekonstruksi_d3)
    
    qrs_on = qrs_set['qrs_on']
    qrs_off = qrs_set['qrs_off']
    #p_waves = func_pwave(data, sinyal_d45, rpeak, fs)
    p_waves = func_pwave(data, sinyal_d45, rpeak, qrs_on, qrs_off, fs)
    
    
    PQRST_wave = {'rpeak': r_peak, 'qrs_onset': qrs_on, 'q_wave': q_waves, 's_wave': s_waves, 'qrs_offset' : qrs_off, 't_wave': t_waves, 'p_wave': p_waves}
    if plot == True:
        plot_graph(data, PQRST_wave)
    return PQRST_wave

def rpeak_correction(signal, r_peak, fs):
    tol = int(0.08 * fs) #toleransi 0.08 (paper DWT noise removal)
    panjang = len(signal)
    
    newR = []
    for r in r_peak:
        awal = r - tol
        if awal < 0:
            continue
        batas = r + tol
        if batas > panjang:
            break
        R = awal + np.argmax(signal[awal:batas]) 
        newR.append(R)
    return newR

#def r_peak_det(signal, d234, )

def func_qwave(dataset, rpeaks, fs):
    idx = int(0.1*fs)
    q_waves = []
    for i in range(len(rpeaks)):        
        a = rpeaks[i]
        b = np.argmin(dataset[a-idx:a])
        q_waves.append(a-(idx-b))
    return q_waves

def func_swave(dataset, rpeaks, fs):
    idx = int(0.1*fs)
    s_waves = []
    for i in range(len(rpeaks)):        
        a = rpeaks[i]
        b = np.argmin(dataset[a:a+idx])
        s_waves.append(a+b)
    return s_waves

def func_qrs_set(sinyal_asli, rpeak, fs, sinyal_d234, sinyal_d3):
    qrs_on = []
    qrs_off = []
    zero_point = zero_cross(sinyal_asli, sinyal_d234, rpeak, fs)
    q_wave = func_qwave(sinyal_asli, rpeak, fs)
    s_wave = func_swave(sinyal_asli, rpeak, fs)
    ## BATAS QRS OFF
    for i in range(len(rpeak)):
        thr = 0.4*sinyal_asli[rpeak[i]] 
        if sinyal_asli[s_wave[i]] >= (thr*-1):
            s_234 = func_swave(sinyal_d234, rpeak, fs)
            a = s_234[i]
            idx = int(0.1*fs)
            b = np.argmax(sinyal_d234[a:a+idx])
            qrs_off.append(a+b)
        elif sinyal_asli[s_wave[i]] <= (thr*-1):
            s_3 = func_swave(sinyal_d3, rpeak, fs)
            a = s_3[i]
            idx = int(0.1*fs)
            b = np.argmax(sinyal_d3[a:a+idx])
            qrs_off.append(a+b)
    ## BATAS QRS OFF
    ## BATAS QRS ON
    idx = int(0.03*fs)
    if zero_point != 0:
        for i in range(len(zero_point)):
            qrs_on.append(zero_point[i])
    for i in q_wave:
        a = i - idx
        qrs_on.append(a)
        qrs_on.sort()
    ## BATAS QRS ON
    qrs = {'q_wave': q_wave, 'qrs_on': qrs_on, 's_wave': s_wave, 'qrs_off': qrs_off}
    return qrs          

def zero_cross(sinyal_asli, sinyal_d234, rpeak, fs):
    idx = int(0.1*fs)
    zero1 = []
    zero2 = []
    cross_point = []
    for i in rpeak:
        a = i
        b = i-idx
        while a!=b:
            if sinyal_asli[a] == 0:
                zero1.append(a)
            elif sinyal_d234[a] == 0:
                zero2.append(a)
            else:
                zero1.append(0)
                zero2.append(1)
            a=a-1
    for i in range(len(zero1)):
        if zero1[i] == zero2[i]:
            cross_point.append(zero2[i])
    print ("NILAI ZERO_CROSSING POINT = ", cross_point) #UNTUK MODIFIKASI KODING
    return cross_point

def func_twave(dataset, rpeaks, fs):
    t_waves = []
    for index, rpeak in enumerate(rpeaks[0:-1]):
       epoch = np.array(dataset)[int(rpeak):int(rpeaks[index+1])] #nyari titik antar r interval
       middle = (rpeaks[index+1] - rpeak) / 2
       quarter = middle/2
       epoch = np.array(dataset)[int(rpeak+quarter):int(rpeak+middle)]
       t_wave = int(rpeak+quarter) + np.argmax(epoch)
       t_waves.append(t_wave)
    return t_waves
        
def func_pwave(sinyal, sinyal_d45, rpeak, qrs_on, qrs_off, fs):
    """p_waves = []
    
    t0 = t_waves.copy()
    t0.insert(0,0)
    list(t0)
    
    for index, awal in enumerate(t0[0:-1]):
        epoch = np.array(dataset)[int(awal):int(rpeaks[index])]
        middle = (rpeaks[index] - int(awal)) / 2 #SAMA DENGAN 60
        quarter = middle/2
        epoch = np.array(dataset)[int(awal):int(rpeaks[index]-quarter)]
        p_wave = int(awal) + np.argmax(epoch)
        p_waves.append(p_wave)"""
    #RRav
    all_rr=[]
    for i in rpeak[1]:
        all_rr.append(rpeak[1][i-1]-rpeak[1][i])
    rr_av=np.mean(all_rr)
    #Perbandingan RRav
    bat1=1.5*rr_av
    bat2=0.5*rr_av
    rr_av_list=[]
    for i in rpeak[1]:
        if bat1>rpeak[1][i-1] and rpeak[1][i-1]>bat2:
            rr_av_new=0.8*rr_av+0.2*rpeak[1][i-1]
        else:
            rr_av_new=rr_av
        rr_av_list.append(rr_av_new)
    """rr_list = []
    for i in range(20):
        rr_list.append(rpeak[i+1] - rpeak[i])
    rr = np.mean(rr_list)
    """
    rr_av20l=[]
    for i in range(20):
        rr_av20l.append(rr_av_list[i-1])
    rr_av20=np.mean(rr_av20l)
    OSW_list=[]
    p_wave=[]
    #batas_kiri = []
    #batas_kanan = []
    for i in range(len(rpeak)-1):
        #OSW section
        a = int(qrs_on[i]-0.33*rr_av20*fs)
        b = int(qrs_on[i]-15)
        OSW = np.array(sinyal[a:b])  #OSW UNTUK DENOISED SIGNAL
        OSW_d45 = np.array(sinyal_d45[a:b])  #OSW UNTUK SINYAL REKONSTRUKSI D45
        OSW_list.append(OSW_d45)

        # THRESHOLDING
        a1 = a-20
        b1 = b+40
        n = sinyal_d45[a1:b1]
        maks = np.argmax(n)+a1
        print(maks)
        thr = 0.125*(abs(sinyal_d45[maks]))
        Eps_p=abs(maks-qrs_on)/rr_av20
        if Eps_p<0.12:
            a=int(qrs_on-0.18*rr_av20*fs)
            b=int(qrs_on-0.12*rr_av20*fs)
            NSW=np.array(sinyal[a:b]) #NSW DENGAN DENOISED SIGNAL
            p_peak = np.argmax(NSW)+(a1)
        elif thr>maks:
            p_peak = np.argmax(OSW)+(a1)  #p_peak=maks value in denoised data
        else:
            p_peak = np.argmax(OSW_d45)+(a1)  #p_peak=maks value in D45
        p_wave.append(p_peak)
        #p_min_d45.append(p_end)
        #batas_kiri.append(a)
        #batas_kanan.append(b)
    #hasil = {'index_max45': p_max_d45, 'kiri': batas_kiri, 'kanan': batas_kanan}
    return p_wave    

def plot_graph(dataset, waves):
    rpeak = waves['rpeak']
    qrs_on = waves['qrs_onset']
    qrs_off = waves['qrs_offset']
    q = waves['q_wave']
    s = waves['s_wave']
    t = waves['t_wave']
    p = waves['p_wave']
    
    plt.figure("Feature Extraction (PQRST PLOT)")
    plt.plot(dataset, color='#1052ba')
    plt.plot(dataset, markevery=rpeak, marker="o", color='blue', markersize=6, linestyle='None')
    plt.plot(dataset, markevery=qrs_on, marker="o", color='red', markersize=6, linestyle='None')
    plt.plot(dataset, markevery=q, marker="o", color='#f95ed0', markersize=6, linestyle='None')
    plt.plot(dataset, markevery=s, marker="o", color='green', markersize=6, linestyle='None')
    plt.plot(dataset, markevery=qrs_off, marker="o", color='purple', markersize=6, linestyle='None')
    plt.plot(dataset, markevery=p, marker="o", color='#e8a302', markersize=6,linestyle='None')
    plt.plot(dataset, markevery=t, marker="o", color='black', markersize=6, linestyle='None')
    #Legend
    r_lgn = mlines.Line2D([], [], color='blue', marker='o', markersize=10, label='R Peak', linestyle="None")
    p_lgn = mlines.Line2D([], [], color='#e8a302', marker='o', markersize=10, label='P Waves', linestyle="None")
    on_lgn = mlines.Line2D([], [], color='red', marker='o', markersize=10, label='QRS Onset', linestyle="None")
    q_lgn = mlines.Line2D([], [], color='#f95ed0', marker='o', markersize=10, label='Q Waves', linestyle="None")
    s_lgn = mlines.Line2D([], [], color='green', marker='o', markersize=10, label='S Waves', linestyle="None")
    off_lgn = mlines.Line2D([], [], color='purple', marker='o', markersize=10, label='QRS Offset', linestyle="None")
    t_lgn = mlines.Line2D([], [], color='black', marker='o', markersize=10, label='T Waves', linestyle="None")
    plt.legend(handles=[r_lgn,on_lgn,q_lgn,s_lgn,off_lgn,p_lgn,t_lgn])
    plt.show()