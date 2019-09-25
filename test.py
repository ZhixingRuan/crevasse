import ctypes
import glob
import time

import numpy as np
from ckmeans import ckmeans, ckmeans_predict
from matplotlib import pyplot as plt
from numpy.ctypeslib import ndpointer
from PIL import Image
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


ll = ctypes.cdll.LoadLibrary
lss = ll('./ssdesc.so')
# function ssdesc_calc return ssdescs -- array of self-similarity descriptors


# ---------------------------------------------------------------------------
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

# parms = Parms(5, 40, 3, 12, 300000, 0.7, 0.7, 0.85)
angbin = 12
radbin = 3
parms = Parms(3, 20, radbin, angbin, 300000, 0.7, 0.7, 0.85)
#parms = Parms(5, 40, radbin, angbin, 300000, 0.7, 0.7, 0.85)
# -------------------------------------------------------------------------
# calculate descriptors for samples

def cal_descriptor(filepath, type_name):
    # type_name = 1 is crevasse
    # type_name = 0 is non_crevasse
    # type_name = -1 is testing images

    descriptor = []
    file_name = []

    for filename in glob.glob(filepath + '*.tif'):
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
        ssdesc_calc(
            img.flatten(order='F'), width, height, channels, parms, ssdescs_flat
        )

        ssdescs = ssdescs_flat.reshape(tuple(reversed(N)), order='F')
        descriptor.append(ssdescs)

        filename = filename.replace(filepath, '')
        filename = filename.replace('.tif', '')
        file_name.append(int(filename))
        
    file_list = np.zeros((len(file_name), 2))
    for i in range(len(file_name)):
        file_list[i][0] = file_name[i]
        file_list[i][1] = type_name

    return [descriptor,file_list]


def descriptor_norm(descriptor):    
    des = []
    ang, rad, row, col = descriptor.shape
    for i in range(0, row):
        for j in range(0, col):
            lss = descriptor[:, :, i, j]
            lss = lss.flatten()
            des.append(lss)
    des = np.array(des)

    desmax = max(des.flatten())
    desmin = min(des.flatten())
    des_norm = []
    for i in range(0, len(des)):
        desnorm = (des[i] - desmin) / (desmax - desmin)
        des_norm.append(desnorm)
    des_norm = np.array(des_norm)

    return des_norm

def prep_descriptor(descriptor, filelist):
    des_list = []
    img_list = []
    type_list = []
    for i in range(0, len(descriptor)):
        d = np.array(descriptor[i])
        for j in range(0, len(d)):
            des_list.append(d[j])
            img_list.append(filelist[i][0])
            type_list.append(filelist[i][1])
    des_list = np.array(des_list)

    file_list = np.zeros((len(img_list), 2))
    for i in range(len(img_list)):
        file_list[i][0] = img_list[i]
        file_list[i][1] = type_list[i]

    return [des_list, file_list]

# -----------------------------------------------------------------------------------
# bag of visual language for clusters in each image

def bov_clusters(cluster_label, img_list, n_clusters, n_data, n_cr):
    #hist = defaultdict(lambda: np.zeros(n_clusters))
    hist = np.array([np.zeros(n_clusters) for i in range(n_data)])
    for i in range(0, len(cluster_label)):
        if img_list[i][1] == 1:
            index = int(img_list[i][0])
            cluster = int(cluster_label[i])
            hist[index][cluster] += 1
        elif img_list[i][1] == 0:
            index = n_cr + int(img_list[i][0])
            cluster = int(cluster_label[i])
            hist[index][cluster] += 1
        elif img_list[i][1] == -1:
            index = int(img_list[i][0])
            cluster = int(cluster_label[i])
            hist[index][cluster] += 1

    return hist

def hist_plot(hist, n_clusters, row, col):
    plt.figure()
    for i in range(0, len(hist)):
        plt.subplot(row, col, i+1)
        plt.bar(np.arange(n_clusters), hist[i])

# -----------------------------------------------------------------------------------
if __name__ == '__main__':
     
    cr_path = 'sample/cr_2/'
    bg_path = 'sample/bg_2/' 
    channels = 1
    test_path = 'sample/test/'

    descriptor_cr, filelist_cr = cal_descriptor(cr_path, type_name = 1)
    descriptor_bg, filelist_bg = cal_descriptor(bg_path, type_name = 0)
    descriptor = descriptor_cr + descriptor_bg
    filelist = np.append(filelist_cr, filelist_bg, axis=0)
    
    descriptor_n = []
    for i in range(0, len(descriptor)):
        descriptor_n.append(descriptor_norm(descriptor[i]))

    sample_des, sample_list = prep_descriptor(descriptor_n, filelist)

    n_clusters = 100
    iteration = 200
    n_data = len(descriptor_n)
    n_cr = len(descriptor_cr)
    n_bg = len(descriptor_bg)
    
    # cluster sample features (slow)
    sample_label, label_convergence, centroids, centers_shift = ckmeans(data=sample_des, n_clusters=n_clusters, max_iterations=iteration, seed=0, rtol=1e-4)
    # save temporary data
    #np.save('sample_label', sample_label)
    #np.save('centroids', centroids)

    
    # read temporary data
    #centroids = np.load('centroids.npy')
    # read txt
    #sample_label = np.loadtxt('sample_label.txt')
    
    hist = bov_clusters(sample_label, sample_list, n_clusters, n_data, n_cr)
    #standardization of histogram
    scale = StandardScaler().fit(hist)
    hist_std = scale.transform(hist)

    hist_plot(hist_std, n_clusters, 11, 8)
    

    
    # processing test images
    descriptor_test, filelist_test = cal_descriptor(test_path, type_name = -1)
    descriptor_n_test = []
    for i in range(0, len(descriptor_test)):
        descriptor_n_test.append(descriptor_norm(descriptor_test[i]))
    
    n_test = len(descriptor_n_test)
    test_des, test_list = prep_descriptor(descriptor_n_test, filelist_test)
    test_label = ckmeans_predict(data=test_des, centroids=centroids)

    hist_test = bov_clusters(test_label, test_list, n_clusters, n_test, n_cr)
    hist_std_test = scale.transform(hist_test)
    
    # train model
    #cv = KFold(n_splits=10, random_state=42, shuffle=False)
    sample_y = filelist[:,1]
    sample_x = hist_std
    model = XGBClassifier()    
    scores = cross_val_score(model, sample_x, sample_y, cv=10)
    
    '''
    model.fit(sample_x, sample_y)
    y_pred = model.predict(hist_std_test)
    y_test = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1])

    mis = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            mis.append(int(filelist_test[i][0]))
    '''

    
    


   
   
    ''' 
    # calculate descriptors for one image----for test
    filename = 'sample/cr_2/11.tif'
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
            lss = ssdescs[:, :, i, j]
            lss = lss.flatten()
            lssDescriptor.append(lss)
    lssDescriptor = np.array(lssDescriptor)

    lss_norm = []
    for i in range(0, len(lssDescriptor)):
        lssmax = max(lssDescriptor.flatten())
        lssmin = min(lssDescriptor.flatten())
        lssnorm = (lssDescriptor[i] - lssmin) / (lssmax - lssmin)
        lss_norm.append(lssnorm)
    lss_norm = np.array(lss_norm)
    '''
    '''
    start = time.time()
    labels, label_convergence, centroids, centers_shift = ckmeans(
        data=lss_norm, n_clusters=10, max_iterations=200, seed=0
    )
    print(f'Processing time: {time.time() - start} seconds')

    plt.plot(label_convergence)
    plt.show()
    '''
    

