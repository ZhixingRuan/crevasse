from collections import defaultdict

import numpy as np
from numpy.fft import fft, ifft
from pathos.multiprocessing import ProcessingPool as Pool

POOL = Pool(nodes=8)


class CKMEANS:
    def __init__(self, data, n_clusters, max_iterations, seed=0):
        self.seed = seed

        self.n_clusters = n_clusters

        self.max_iterations = max_iterations
        self.total_iterations = 0
        self.label_convergence = []

        self.data = data
        self.centroids = None
        self.labels = np.empty(data.shape[0])

    def compute_distance(self, point, centroid):
        r = ifft(fft(centroid) * fft(point).conj()).real
        r = r[::3]
        distance = -2 * max(r)
        distance += np.dot(point, point)
        distance += np.dot(centroid, centroid)

        rotation_index = np.argmax(r)

        return distance, rotation_index

    def assign_to_cluster(self, point):
        distances, rotation_indices = zip(
            *[self.compute_distance(point, centroid) for centroid in self.centroids]
        )
        closest_cluster = np.argmin(distances)
        rotation_index = rotation_indices[closest_cluster]
        point_shift = np.roll(point.reshape(8, 3), rotation_index)
        point_shift = point_shift.flatten()

        return closest_cluster, point_shift

    def init_centroids(self):
        k = self.n_clusters
        n_samples = self.data.shape[0]
        random_state = np.random.RandomState(self.seed)
        seeds = random_state.permutation(n_samples)[:k]
        centers = self.data[seeds]
        return np.array(centers)

    def perform_iteration(self):
        cluster_points = defaultdict(list)
        # First assign each point to the nearest cluster
        for point_idx, point in enumerate(self.data):
            cluster_idx, point_shift = self.assign_to_cluster(point)
            cluster_points[cluster_idx].append(point_shift)
            self.labels[point_idx] = cluster_idx

        # Now update the centroids based on the new cluster assignments
        for cluster_idx, shifted_points in cluster_points.items():
            self.centroids[cluster_idx] = np.mean(shifted_points, axis=0)

    def run(self):
        self.centroids = self.init_centroids()
        while self.total_iterations < self.max_iterations:
            previous_centroids = self.centroids.copy()
            previous_labels = self.labels.copy()

            self.perform_iteration()

            self.label_convergence.append(
                sum(np.array(previous_labels) != np.array(self.labels))
            )

            if np.allclose(previous_centroids, self.centroids):
                break

            self.total_iterations += 1


class CKMEANS_MULTI(CKMEANS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform_iteration(self):
        # First assign each point to the nearest cluster
        results = zip(*POOL.map(self.assign_to_cluster, enumerate(self.data)))
        cluster_points = defaultdict(list)
        for point_idx, (cluster_idx, point_shift) in enumerate(results):
            cluster_points[cluster_idx].append(point_shift)
            self.labels[point_idx] = cluster_idx

        # Now update the centroids based on the new cluster assignments
        for cluster_idx, shifted_points in cluster_points.items():
            self.centroids[cluster_idx] = np.mean(shifted_points, axis=0)
