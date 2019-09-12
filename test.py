import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from itertools import permutations, product
from matplotlib import pyplot as plt
from PIL import Image
import glob
from numpy.fft import fft, ifft
from numpy.linalg import norm
from collections import defaultdict
import pdb
from ckmeans import *
from ckmeans_multi import *
from pathos.multiprocessing import ProcessingPool as Pool

ll = ctypes.cdll.LoadLibrary
lss = ll('./ssdesc.so')
# function ssdesc_calc return ssdescs -- array of self-similarity descriptors

#---------------------------------------------------------------------------
class Parms(ctypes.Structure):
    _fields_ = [
        ('patch_size', ctypes.c_ushort),
        ('cor_size', ctypes.c_ushort),
        ('nrad', ctypes.c_ushort),
        ('nang', ctypes.c_ushort),
        ('var_noise', ctypes.c_double),
        ('saliency_thresh', ctypes.c_double),
        ('homogeneity_thresh', ctypes.c_double),
        ('snn_thresh', ctypes.c_double),
    ]

ssdesc_calc = lss.ssdesc_calc
ssdesc_calc.argtypes = [
    ndpointer(np.double, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.Structure,
    ndpointer(np.double, flags='C_CONTIGUOUS'),
]

#parms = Parms(5, 40, 3, 12, 300000, 0.7, 0.7, 0.85)
angbin = 8
radbin = 3
parms = Parms(3, 20, radbin, angbin, 200000, 0.7, 0.7, 0.85)
#-------------------------------------------------------------------------
#calculate descriptors for samples

def cal_descriptor(filepath):
    descriptor = []
    for filename in glob.glob(filepath):
        img = np.asarray(Image.open(filename), dtype=np.double)
        height, width = img.shape
        # initialize output array to pass into c function
        N = (
            (height - parms.cor_size + 1),
            (width - parms.cor_size + 1),
            parms.nrad,
            parms.nang,
            )
        n = np.product(N)
        ssdescs_flat = np.zeros(n, dtype=np.double)
        # order ='F' for column major
        ssdesc_calc(img.flatten(order='F'), width, height, channels, parms, ssdescs_flat)

        ssdescs = ssdescs_flat.reshape(tuple(reversed(N)), order='F')
        descriptor.append(ssdescs)

    return descriptor

def descriptor_norm(descriptor):
    lss_norm = []
    for i in range(0, len(descriptor)):
        lssmax = max(descriptor.flatten())
        lssmin = min(descriptor.flatten())
        lssnorm = (descriptor[i] - lssmin)/(lssmax - lssmin)
        lss_norm.append(lssnorm)

    return lss_norm

#-----------------------------------------------------------------------------------
if __name__ == '__main__':

    #calculate descriptors for samples
    filename = 'sample/cr/11.tif'
    channels = 1
    img = np.asarray(Image.open(filename), dtype=np.double)
    height, width = img.shape
    # initialize output array to pass into c function
    N = (
        (height - parms.cor_size + 1),
        (width - parms.cor_size + 1),
        parms.nrad,
        parms.nang,
        )
    n = np.product(N)
    ssdescs_flat = np.zeros(n, dtype=np.double)
    # order ='F' for column major
    ssdesc_calc(img.flatten(order='F'), width, height, channels, parms, ssdescs_flat)

    ssdescs = ssdescs_flat.reshape(tuple(reversed(N)), order='F')
    
    ang, rad, row, col = ssdescs.shape
    lssDescriptor = []
    for i in range(0, row):
        for j in range(0, col):
            lss = ssdescs[:,:,i,j]
            lss = lss.flatten()
            lssDescriptor.append(lss)
    #import ipdb; ipdb.set_trace()
    lssDescriptor = np.array(lssDescriptor)

    lss_norm = []
    for i in range(0, len(lssDescriptor)):
        lssmax = max(lssDescriptor.flatten())
        lssmin = min(lssDescriptor.flatten())
        lssnorm = (lssDescriptor[i]-lssmin)/(lssmax-lssmin)
        lss_norm.append(lssnorm)
    lss_norm = np.array(lss_norm)

    kmeans = CKMEANS(data=lss_norm, nclusters=10, iteration=200, randomstate=0)
    kmeans.k_means()
    change = kmeans.change
    plt.plot(change)
    
    kmeans_multi = CKMEANS_MULTI(data=lss_norm, nclusters=10, iteration=200, randomstate=0)
    kmeans_multi.k_means())
