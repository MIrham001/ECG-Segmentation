data = wfdb.rdrecord('/home/han/S.Kom/Database/qtdb/sel100')
data_dictionary = data.__dict__

signal = data_dictionary['p_signal']
signal_ml2 = signal[:,0]

a = wfdb.rdann('/home/han/S.Kom/Database/qtdb/sel100','pu0')
a_dictionary = a.__dict__
ann = a_dictionary['sample']
#DATA2 = wfdb.plot_items(signal_ml2, [ann])

############# BATAS DAPETIN NILAI T #############
t_idx = a_dictionary['sample']
t_val = a_dictionary['symbol']
tanda = []
t_loc =[]
for i in range(len(t_val)):
    if t_val[i] == 't':
        tanda.append(1)
        t_loc.append(t_idx[i])
    else:
        tanda.append(np.nan)
        t_loc.append(np.nan)
        
for i in range(len(t_val)):
    if t_val[i] == 't':
        if t_val[i-1] == '(':
            tanda[i-1] = 'on'
            t_loc[i-1] = t_idx[i-1]
        if t_val[i+1] == ')':
            tanda[i+1] = 'off'
            t_loc[i+1] = t_idx[i+1]
x = {'t_idx': a_dictionary['sample'], 't_val': a_dictionary['symbol'], "label": tanda, "t_loc": t_loc}
anotasi = pd.DataFrame(x)
anotasi_t = anotasi.dropna()
z = np.array(anotasi_t['t_loc'])
z = z.astype(int)
#DATA1 = wfdb.plot_items(new_signal, [z])
DATA2 = wfdb.plot_items(signal_ml2, [z])
