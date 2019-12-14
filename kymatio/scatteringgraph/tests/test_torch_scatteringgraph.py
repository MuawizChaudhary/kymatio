import pytest
import numpy as np
import torch
from kymatio.scatteringgraph.utils import compute_degree_vector

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


backends = []
backends_devices = []

from kymatio.scatteringgraph.backend.torch_backend import backend
backends.append(backend)
backends_devices.append((backend, 'cpu'))

if 'cuda' in devices:
    backends_devices.append((backend, 'cuda'))

class TestMoment:
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_moment(self, backends_devices):
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)

        backend, device = backends_devices

        A = torch.from_numpy(A)
        x = torch.from_numpy(degree_vector_A).to(device)

        #q = 1
        mean = np.array([[1]])
        mean = torch.from_numpy(mean).float()

        mean_moment = backend.moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[0]])
        var = torch.from_numpy(var).float()

        var_moment = backend.moment(x, 2, mean=mean_moment)
        assert torch.allclose(var, var_moment)

        #q = 3
        #come up with a better example than cliques
               
        # three clique adjacency matrix
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)

        x = torch.from_numpy(degree_vector_A).to(device)

        #q = 1
        mean = np.array([[2]])
        mean = torch.from_numpy(mean).float()

        mean_moment = backend.moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[0]])
        var = torch.from_numpy(var).float()

        var_moment = backend.moment(x, 2, mean=mean_moment)
        assert torch.allclose(var, var_moment)

        #q = 3
        #come up with a better example than cliques
