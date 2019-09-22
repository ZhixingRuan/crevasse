from collections import defaultdict
from functools import partial

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.fftpack import fft, ifft


def ckmeans(data, n_clusters, max_iterations, seed=0):
    total_iterations = 0
    label_convergence = []
    labels = np.empty(data.shape[0])

    centroids = init_centroids(data, n_clusters)
    while total_iterations < max_iterations:
        previous_centroids = centroids.copy()
        previous_labels = labels.copy()

        # labels, centroids = perform_iteration(data, centroids)
        labels, centroids = perform_iteration_multi(data, centroids)

        label_convergence.append(sum(np.array(previous_labels) != np.array(labels)))

        if np.allclose(previous_centroids, centroids):
            break

        total_iterations += 1

    return labels, label_convergence


def compute_distance(point, centroid):
    r = ifft(fft(centroid) * fft(point).conj()).real
    r = r[::3]
    distance = -2 * max(r)
    distance += np.dot(point, point)
    distance += np.dot(centroid, centroid)

    rotation_index = np.argmax(r)

    return distance, rotation_index


def assign_to_cluster(point, centroids):
    distances, rotation_indices = zip(
        *[compute_distance(point, centroid) for centroid in centroids]
    )
    closest_cluster = np.argmin(distances)
    rotation_index = rotation_indices[closest_cluster]
    point_shift = np.roll(point.reshape(8, 3), rotation_index)
    point_shift = point_shift.flatten()

    return closest_cluster, point_shift


def init_centroids(data, n_clusters, seed=0):
    k = n_clusters
    n_samples = data.shape[0]
    random_state = np.random.RandomState(seed)
    seeds = random_state.permutation(n_samples)[:k]
    centers = data[seeds]

    return np.array(centers)


def perform_iteration(data, centroids):
    labels = np.empty(data.shape[0])
    cluster_points = defaultdict(list)
    # First assign each point to the nearest cluster
    for point_idx, point in enumerate(data):
        cluster_idx, point_shift = assign_to_cluster(point, centroids)
        cluster_points[cluster_idx].append(point_shift)
        labels[point_idx] = cluster_idx

    # Now update the centroids based on the new cluster assignments
    for cluster_idx, shifted_points in cluster_points.items():
        centroids[cluster_idx] = np.mean(shifted_points, axis=0)

    return labels, centroids


def perform_iteration_multi(data, centroids):
    # First assign each point to the nearest cluster mapping points on pool of processes
    func = partial(assign_to_cluster, centroids=centroids)
    results = Pool().map(func, data)

    labels = np.empty(data.shape[0])
    cluster_points = defaultdict(list)
    for point_idx, (cluster_idx, point_shift) in enumerate(results):
        cluster_points[cluster_idx].append(point_shift)
        labels[point_idx] = cluster_idx

    # Now update the centroids based on the new cluster assignments
    for cluster_idx, shifted_points in cluster_points.items():
        centroids[cluster_idx] = np.mean(shifted_points, axis=0)

    return labels, centroids
