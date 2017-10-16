from numpy.fft import fft, ifft
import numpy as np

def decode(a, b):
    return ifft(fft(a) * fft(b).conj()).real / len(a)

def encode(a, b):
    return np.real(ifft(fft(a)*fft(b)))

def embed_2d(d, distr, param = [1, 1]):
    rand = np.random.rand(d)
    spike = list(map(lambda x: round(float(bool(x < param[1]))), rand))
    slab = np.random.normal(size=d) * param[1]
    return np.multiply(spike, slab)

def normalize(a):
    return a/np.linalg.norm(a)