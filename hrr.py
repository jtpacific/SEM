from numpy.fft import fft, ifft
import numpy as np

'''
currently normalizing all vectors to train and evaluate neural networks using MSE
decode, encode, embed_2d functions do not require normalization when training and evaluating using cosine proximity

'''

def decode(a, b):
    return normalize(ifft(fft(a) * fft(b).conj()).real / len(a))

def encode(a, b):
    return normalize(np.real(ifft(fft(a) * fft(b))))

# embed symbols as gaussian random vectors
def embed_2d(d):
    return normalize(np.random.normal(0, 1./np.sqrt(d), d))

def normalize(a):
    return a/np.linalg.norm(a)