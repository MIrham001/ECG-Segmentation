import pywt
import numpy as np

def dwt(signal):
    levels = 8

    a = signal
    w = pywt.Wavelet('sym5')
    ca = []
    cd = []
    
    d234 = []

    for level in range(levels):
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)
    a = [0] * len (a)
    a = np.array(a)
    cd.append(a)

    for i,d in enumerate(cd):
        if i in [1,2,3]:
            d234.append(cd[i])
    
    from statsmodels.robust import mad
    
    sigma = mad( cd[0] ) #NILAI dari to (to = median(cD1))
    uthresh = sigma * np.sqrt(2*np.log( len(signal))) # rumus threshold (thr = akar(2 log N))

    new_cd = []
    for d in cd:
        new_cd.append(pywt.threshold(d, value=uthresh, mode="soft"))
    
    new_cd.reverse()
    new_signal = pywt.waverec(new_cd, w)
    D2 = wrcoef(signal, 'd', new_cd, w, 2)
    D3 = wrcoef(signal, 'd', new_cd, w, 3)
    D4 = wrcoef(signal, 'd', new_cd, w, 4)
    D5 = wrcoef(signal, 'd', new_cd, w, 5)
    D6 = wrcoef(signal, 'd', new_cd, w, 6)
    wavelet = {'sinyal': new_signal, 'D2': D2, 'D3': D3, 'D4': D4, 'D5': D5, 'D6': D6}
    return wavelet

def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))
