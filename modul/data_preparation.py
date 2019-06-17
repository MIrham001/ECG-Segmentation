import numpy as np

def load(lokasi_file, nama_file):
    load_data = np.loadtxt(lokasi_file+ str(nama_file), delimiter=',')
    return load_data

def save(data, lokasi_file, nama_file):
    save_data = np.savetxt(lokasi_file + str(nama_file), data, fmt='%0.3f', delimiter=',')
    return save_data
