import pytest
import numpy as np
import torch
from kymatio.scatteringgraph.utils import compute_degree_vector
from kymatio.scatteringgraph import ScatteringGraph

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
    def test_normalized_moment(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)

        moment = backend.normalized_moment 

        A = torch.from_numpy(A)
        x = torch.from_numpy(degree_vector_A).to(device).double()

        #q = 1
        mean = np.array([[1]])
        mean = torch.from_numpy(mean).double()

        mean_moment = moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[0]])
        var = torch.from_numpy(var).double()

        var_moment = moment(x, 2, mean=mean_moment)
        assert torch.allclose(var, var_moment)

        #q = 3
        #come up with a better example than cliques
               
        # three clique adjacency matrix
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)

        x = torch.from_numpy(degree_vector_A).to(device).double()

        #q = 1
        mean = np.array([[2]])
        mean = torch.from_numpy(mean).double()

        mean_moment = moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[0]])
        var = torch.from_numpy(var).double()

        var_moment = moment(x, 2, mean=mean_moment)
        assert torch.allclose(var, var_moment)

        #q = 3
        #come up with a better example than cliques

    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_unnormalized_moment(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)

        moment = backend.unnormalized_moment 

        A = torch.from_numpy(A)
        x = torch.from_numpy(degree_vector_A).to(device).double()

        #q = 1
        mean = np.array([[2]])
        mean = torch.from_numpy(mean).double()

        mean_moment = moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[2]])
        var = torch.from_numpy(var).double()

        var_moment = moment(x, 2)
        assert torch.allclose(var, var_moment)

        #q = 3
        #come up with a better example than cliques
               
        # three clique adjacency matrix
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)

        x = torch.from_numpy(degree_vector_A).to(device).double()

        #q = 1
        mean = np.array([[6]])
        mean = torch.from_numpy(mean).double()

        mean_moment = moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[12]])
        var = torch.from_numpy(var).double()

        var_moment = moment(x, 2)
        assert torch.allclose(var, var_moment)

        #q = 3
        #come up with a better example than cliques

class TestScatteringGraph:
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_scattering_graph(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])

        S = ScatteringGraph(J=2, Q=2, A=A, normalize=False, max_order=2,
                backend=backend)
        S = S.to(device)

        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()

        S_x = S(x)
        print(S_x.shape)
        print(S_x)
        assert False



        

