import pytest
import numpy as np
from kymatio.scatteringgraph.utils import compute_degree_vector

class TestComputeDegreeVector:
    def test_compute_degree_vector(self):
        # node A direct connection to node B
        A_1 = np.array([[0, 1], [0, 0]])
        deg_A_1 = np.array([1, 0]).reshape(-1, 1)

        degree_vector_A_1 = compute_degree_vector(A_1)

        assert np.allclose(degree_vector_A_1, deg_A_1)

        # two clique adjacency matrix
        A_2 = np.array([[0, 1], [1, 0]])
        deg_A_2 = np.array([1,1]).reshape(-1, 1)

        degree_vector_A_2 = compute_degree_vector(A_2)

        assert np.allclose(degree_vector_A_2, deg_A_2)
        
        # three clique adjavency matrix  
        A_3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        deg_A_3 = np.array([2, 2, 2]).reshape(-1, 1)

        degree_vector_A_3 = compute_degree_vector(A_3)

        assert np.allclose(degree_vector_A_3, deg_A_3)
