import numpy as np


def compute_degree_vector(A):
    "Computes the degree of each node from a weighted adjacency matrix"
    return np.sum(A, axis=1).reshape(-1, 1)

