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
from tqdm import tqdm

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

    for index_centroid in range(0, total_cen):
        new_centroids[index_centroid] = np.mean(label_points[index_centroid], axis=0)

    return [new_centroids, label]

def k_means(data_points, centroids, iteration):
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    change = []
    
    new_centroids, label = cal_centroids(data_points, centroids)
    #count = 0
    for count in tqdm(range(iteration)):
        #if np.array_equal(new_centroids, centroids) == False:
        if np.allclose(new_centroids, centroids) == False:
            centroids = new_centroids.copy()
            new_centroids, new_label = cal_centroids(data_points, new_centroids)
            change.append(sum(np.array(new_label) != np.array(label)))
            label = new_label.copy()
        else:
            print(count)
            break

    for index_point in range(0, total_points):
        distance = {}
        for index_centroid in range(0, k):
            distance[index_centroid] = compute_euclidean_distance(data_points[index_point], new_centroids[index_centroid])
        index_of_minimum, point_shift, data_point = assign_label_cluster(distance, data_points[index_point], new_centroids)
        #cluster_label.append([index_of_minimum, data_point])
        cluster_label.append(index_of_minimum)
    
    return [cluster_label, change, new_centroids]


def create_centroids(X, random_state, n_clusters):
    k = n_clusters
    n_samples = X.shape[0]
    random_state = np.random.RandomState(random_state)
    seeds = random_state.permutation(n_samples)[:k]
    centers = X[seeds]
    return np.array(centers)

def kmeans_descriptorlist(n_cluster, descriptor, iteration):
    lss_list = []
    img_list = []
    for i in range(0, len(descriptor)):
        d = np.array(descriptor[i])
        row = d.shape[2]
        col = d.shape[3]
        for r in range(0, row):
            for c in range(0, col):
                lss_list.append(d[:,:,r,c].flatten())
                img_list.append(i)

    lss_list = np.array(lss_list)
    centroids = create_centroids(lss_list, 0, n_cluster)
    #pdb.set_trace()
    cluster_label, change, centroids_kmeans = k_means(lss_list, centroids, iteration)

    return [cluster_label, change, img_list]
 
#--------------------------------------------------------------------------------
#bov for clusters in each image
def bov_clusters(cluster_label, img_list, descriptor, n_cluster):
    hist = defaultdict(lambda: np.zeros(n_cluster))
    for i in range(0, len(cluster_label)):
        index = img_list[i]
        cluster = cluster_label[i]
        hist[index][cluster] += 1

    return hist


#--------------------------------------------------------------------------------
cr_path = 'sample/cr/*.tif'
#bg_path = 'sample/bg/*.tif'
channels = 1

descriptor_cr = cal_descriptor(cr_path)
#descriptor_bg = cal_descriptor(bg_path)

descriptor_n = []
for i in range(0, len(descriptor_cr)):
    descriptor_n.append(descriptor_norm(descriptor_cr[i]))

n_clusters = 10
iteration = 200
cluster_label, change, img_list = kmeans_descriptorlist(n_clusters, descriptor_n, iteration)

