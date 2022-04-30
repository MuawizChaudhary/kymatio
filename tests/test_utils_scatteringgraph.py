import pytest
import numpy as np
from kymatio.scatteringgraph.utils import compute_degree_vector


class TestComputeDegreeVector:
    def test_compute_degree_vector(self):
        # node A direct connection to node B
        A = np.array([[0, 1], [0, 0]])
        deg_A = np.array([1, 0]).reshape(-1, 1)

        degree_vector_A = compute_degree_vector(A)

        assert np.allclose(degree_vector_A, deg_A)

        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        deg_A = np.array([1,1]).reshape(-1, 1)

        degree_vector_A = compute_degree_vector(A)

        assert np.allclose(degree_vector_A, deg_A)
        
        # three clique adjavency matrix  
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        deg_A = np.array([2, 2, 2]).reshape(-1, 1)

        degree_vector_A = compute_degree_vector(A)

        assert np.allclose(degree_vector_A, deg_A)

    def test_compute_lazy_walk_matrix(self):
        I2 = np.eye(2)
        I3 = np.eye(3)
        

        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A).reshape(-1,)
        
        D = np.diag(degree_vector_A)
        D_i = np.linalg.inv(D)

        AD_i = np.dot(A, D_i)

        P = np.array([[1/2, 1/2], [1/2, 1/2]])

        assert np.allclose(P, (1/2) * (I2 + AD_i))

        # three clique adjacency matrix
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A).reshape(-1,)
        
        D = np.diag(degree_vector_A)
        D_i = np.linalg.inv(D)

        AD_i = np.dot(A, D_i)

        P = np.array([[1/2, 1/4, 1/4], [1/4, 1/2, 1/4], [1/4, 1/4, 1/2]])

        assert np.allclose(P, (1/2) * (I3 + AD_i))
