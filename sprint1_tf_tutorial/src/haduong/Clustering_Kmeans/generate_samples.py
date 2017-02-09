# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:53:58 2017

@author: Minh Ha Duong
"""

import tensorflow as tf
import numpy as np

from functions import create_samples, choose_random_centroids, plot_clusters, assign_to_nearest, update_centroids

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

model = tf.global_variables_initializer()

# add code here to do multiple iterations

with tf.Session() as session:
    sample_values = session.run(samples)
    updated_centroid_value = session.run(updated_centroids)
    print(updated_centroid_value)

plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)
