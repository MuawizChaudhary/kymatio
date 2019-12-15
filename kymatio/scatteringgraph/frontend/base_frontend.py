from ...frontend.base_frontend import ScatteringBase
#from ..filter_bank import filter_bank
from ..utils import compute_degree_vector
import numpy as np

class ScatteringBaseGraph(ScatteringBase):
    def __init__(self, J, Q=2, A=None, normalize=True, max_order=2, backend='torch'):
        super(ScatteringBaseGraph, self).__init__()
        self.J = J
        self.Q = Q
        self.A = A
        self.normalize = normalize
        self.max_order = max_order
        self.backend = backend

    def build(self):
        "Choose between normalization or no normalization"
        if self.normalize:
            self.backend.moment = self.backend.normalized_moment
        else:
            self.backend.moment = self.backend.unnormalized_moment
        
    def create_filters(self):
        "Create the lazy walk matrix"
        I = np.eye(self.A.shape[0])
        degree_vector_A = compute_degree_vector(self.A).reshape(-1,)
        
        D = np.diag(degree_vector_A)
        D_i = np.linalg.inv(D)

        AD_i = np.dot(self.A, D_i)
        
        P = (1/2) * (I + AD_i)

        self.psi = []
        for j in range(1, self.J + 1):
            P_j_2 = np.power(P, j - 1)
            self.psi.append(np.dot(P_j_2, (I - P_j_2)))

__all__ = ['ScatteringBaseGraph']
