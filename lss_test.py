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

ll = ctypes.cdll.LoadLibrary
lss = ll('./ssdesc.so')
# function ssdesc_calc return ssdescs -- array of self-similarity descriptors
# function ssdesc_norm return resp -- array of normalised descriptors

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
# the last three parms should be 0.7, 0.7, I used 1.0, 1.0 here to disable the homogeneous/salient descriptors detection just for testing
parms = Parms(3, 20, 3, 8, 200000, 1.0, 1.0, 0.85)
#-------------------------------------------------------------------------
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
#ssdescs = np.swapaxes(ssdescs, 2, 3)

#-----------------------------------------------------------------------------------
#kmeans for descriptors
def compute_euclidean_distance(point, centroid):
    r = ifft(fft(centroid) * fft(point).conj()).real
    r = r[::3]
    distance = -2 * max(r)
    distance += np.dot(point, point)
    distance += np.dot(centroid, centroid)
    return distance

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    r = ifft(fft(centroids[index_of_minimum]) * fft(data_point).conj()).real
    r = r[::3]
    rmax_index = np.argmax(r)
    point_shift = np.roll(data_point.reshape(8,3), rmax_index)
    point_shift = point_shift.flatten()
    return [index_of_minimum, point_shift, data_point]

def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    total_points = len(data_points)
    label_points = defaultdict(list)
    new_centroids = []
    k = len(centroids)

    for iteration in range(0, total_iteration):
        cluster_label = []
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
            index_of_minimum, point_shift, data_point = assign_label_cluster(distance, data_points[index_point], centroids)
            label_points[index_of_minimum].append(point_shift)
            cluster_label.append([index_of_minimum, data_point])

        for index_centroid in range(0, k):
            #new_centroids.append(np.mean(label_points[index_centroid]))
            centroids[index_centroid] = np.mean(label_points[index_centroid], axis=0)
    
    return cluster_label

def cal_centroids(data_points, centroids):
    total_points = len(data_points)
    label_points = defaultdict(list)
    total_cen = len(centroids)
    new_centroids = np.zeros(centroids.shape)
    label = []
    
    for index_point in range(0, total_points):
        distance = {}
        for index_centroid in range(0, total_cen):
            distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
        index_of_minimum, point_shift, data_point = assign_label_cluster(distance, data_points[index_point], centroids)
        label_points[index_of_minimum].append(point_shift)
        label.append(index_of_minimum)
        
    #import pdb;pdb.set_trace()
    for index_centroid in range(0, total_cen):
        new_centroids[index_centroid] = np.mean(label_points[index_centroid], axis=0)
    
    return [new_centroids, label]


def k_means(data_points, centroids, iteration):
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    change = []
    
    new_centroids, label = cal_centroids(data_points, centroids)
    count = 0
    while count < iteration:
        #if np.array_equal(new_centroids, centroids) == False:
        if np.allclose(new_centroids, centroids) == False:
            centroids = new_centroids.copy()
            new_centroids, new_label = cal_centroids(data_points, new_centroids)
            change.append(sum(np.array(new_label) != np.array(label)))
            label = new_label.copy()
        else:
            print(count)
            break
        count += 1
    else:
        print("fully iterated")

    for index_point in range(0, total_points):
        distance = {}
        for index_centroid in range(0, k):
            distance[index_centroid] = compute_euclidean_distance(data_points[index_point], new_centroids[index_centroid])
        index_of_minimum, point_shift, data_point = assign_label_cluster(distance, data_points[index_point], new_centroids)
        cluster_label.append([index_of_minimum, data_point])
    
    return [cluster_label, change, new_centroids]


def create_centroids(X, random_state, n_clusters):
    k = n_clusters
    n_samples = X.shape[0]
    random_state = np.random.RandomState(random_state)
    seeds = random_state.permutation(n_samples)[:k]
    centers = X[seeds]
    return np.array(centers)

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

#n_clusters = 200
n_clusters = 10
centroids = create_centroids(lss_norm, 0, n_clusters)
total_iteration = 200
#cluster_label = iterate_k_means(lssDescriptor, centroids, total_iteration)
cluster_label, change, centroids_kmeans = k_means(lss_norm, centroids, total_iteration)
plt.plot(change)

#------------------------------------------------------------------------
#bov for each cluster
data_size = len(cluster_label)
histogram = np.zeros(n_clusters)
for i in range(0, data_size):
    data = cluster_label[i]
    cluster = data[0]
    histogram[cluster] += 1
#normalize histogram
hist_norm = np.zeros(n_clusters)
for i in range(0, n_clusters):
    hist_norm[i] = histogram[i]/sum(histogram)

#plt.plot(histogram)

