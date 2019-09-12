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
from pathos.multiprocessing import ProcessingPool as Pool

class CKMEANS_MULTI:
    def __init__(self, data, nclusters, iteration, randomstate):
        
        """
        randomstate: to create initial centroids
        output: new_centroids -- new centroids after ckmeans
                cluster_label -- list of data and their labels
                count -- time of iteration
        """

        self.n_clusters = nclusters
        self.total_iteration = iteration
        self.random_state = randomstate
        self.data_points = data
        self.total_points = len(data)
        self.init_centroids = np.array([])
        self.new_centroids = np.array([])
        self.temp_centroids = np.array([])
        self.cluster_label = []
        self.change = []
        self.count = 0

    def compute_euclidean_distance(self, point, centroid):
        r = ifft(fft(centroid) * fft(point).conj()).real
        r = r[::3]
        distance = -2 * max(r)
        distance += np.dot(point, point)
        distance += np.dot(centroid, centroid)
        return distance

    def assign_label_cluster(self, distance, data_point, centroids):
        index_of_minimum = min(distance, key=distance.get)
        r = ifft(fft(centroids[index_of_minimum]) * fft(data_point).conj()).real
        r = r[::3]
        rmax_index = np.argmax(r)
        point_shift = np.roll(data_point.reshape(8,3), rmax_index)
        point_shift = point_shift.flatten()
        return [index_of_minimum, point_shift, data_point]

    def cal_centroids(self, centroids):
        label_points = defaultdict(list)
        total_cen = self.n_clusters
        new_centroids = np.zeros(centroids.shape)
        label = []

        for index_point in range(0, self.total_points):
            distance = {}
            data_points = self.data_points[index_point]
            for index_centroid in range(0, total_cen):
                distance[index_centroid] = self.compute_euclidean_distance(data_points, centroids[index_centroid])
            index_of_minimum, point_shift, data_point = self.assign_label_cluster(distance, data_points, centroids)
            label_points[index_of_minimum].append(point_shift)
            label.append(index_of_minimum)

        #import pdb;pdb.set_trace()
        for index_centroid in range(0, total_cen):
            new_centroids[index_centroid] = np.mean(label_points[index_centroid], axis=0)


        return [new_centroids, label]

    def cal_centroids_multi(self, data_point):
        label_points = defaultdict(list)
        total_cen = self.n_clusters
        label = []
        distance = {}
        for index_centroid in range(0, total_cen):
            distance[index_centroid] = self.compute_euclidean_distance(data_point, self.temp_centroids[index_centroid])
        index_of_minimum, point_shift, data_point = self.assign_label_cluster(distance, data_point, centroids)
        label_points[index_of_minimum].append(point_shift)
        label.append(index_of_minimum)

        return [label_points, label]



    def k_means(self):
        k = self.n_clusters
        self.create_centroids()
        centroids = self.init_centroids
        self.temp_centroids, label = self.cal_centroids(self.init_centroids)
        pool = Pool(6)

        while self.count < self.total_iteration:
            if np.allclose(self.temp_centroids, centroids) == False:
                centroids = self.temp_centroids.copy()
                #self.temp_centroids, new_label = self.cal_centroids(self.temp_centroids)
                data = self.data_points
                label_points, new_label = pool.map(self.cal_centroids_multi, data))
                for index_centroid in range(0, k):
                    self.temp_centroids[index_centroid] = np.mean(label_points[index_centroid], axis=0)
                self.change.append(sum(np.array(new_label) != np.array(label)))
                label = new_label.copy()
            else:
                break
            self.count += 1

        for index_point in range(0, self.total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = self.compute_euclidean_distance(self.data_points[index_point], self.temp_centroids[index_centroid])
            index_of_minimum, point_shift, data_point = self.assign_label_cluster(distance, self.data_points[index_point], self.temp_centroids)
            self.cluster_label.append([index_of_minimum, data_point])

        self.new_centroids = self.temp_centroids


    def create_centroids(self):
        k = self.n_clusters
        n_samples = self.data_points.shape[0]
        random_state = np.random.RandomState(self.random_state)
        seeds = random_state.permutation(n_samples)[:k]
        centers = self.data_points[seeds]
        self.init_centroids = np.array(centers)
