import pytest
import numpy as np
import torch
from kymatio.scatteringgraph.utils import compute_degree_vector
from kymatio import ScatteringGraph
from . import  utils
devices = ['cpu'] 
if torch.cuda.is_available():
    devices.append('cuda')

backends = []
backends_devices = []

from kymatio.scatteringgraph.backend.torch_backend import backend
backends.append(backend)
backends_devices.append((backend, 'cpu'))

def gen_P(A):
    I = np.eye(A.shape[0])
    degree_vector_A = compute_degree_vector(A).reshape(-1,)
    
    D = np.diag(degree_vector_A)
    D_i = np.linalg.inv(D)

    AD_i = np.dot(A, D_i)
    
    P = (1/2) * (I + AD_i)
    return P

def gen_W(P):


    psi = []

    I = np.eye(P.shape[0])
    for j in [1, 2, 4, 8, 16]:
        P_j_2 = np.linalg.matrix_power(P, j )
        psi.append(np.matmul(P_j_2, (I - P_j_2)))
    return psi


if 'cuda' in devices:
    backends_devices.append((backend, 'cuda'))
class TestCreateFilters:
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_gen_P(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        P = utils.lazy_random_walk(A)
        P_S = gen_P(A)
        assert P.shape == P_S.shape
        assert np.allclose(P, P_S)
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        P = utils.lazy_random_walk(A)
        P_S = gen_P(A)
        assert P.shape == P_S.shape
        assert np.allclose(P, P_S)

    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_gen_W(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        W_S = gen_W(gen_P(A))
        for i in range(len(W)):
            W_i = W[i]
            W_S_i = W_S[i]
            assert W_i.shape == W_S_i.shape
            assert np.allclose(W_i, W_S_i)
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        W_S = gen_W(gen_P(A))
        for i in range(len(W)):
            W_i = W[i]
            W_S_i = W_S[i]
            assert W_i.shape == W_S_i.shape
            assert np.allclose(W_i, W_S_i)


class TestOrderMoments:
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_zero_order_moments(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=False, max_order=0,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.zero_order_feature(degree_vector_A)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
    

        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=False, max_order=0,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.zero_order_feature(degree_vector_A)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
 
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_first_order_moments(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=False, max_order=1,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.zero_order_feature(degree_vector_A)
        Sx_1 = utils.first_order_feature(np.matmul(W, degree_vector_A))
        Sx_0 = np.concatenate((Sx_0,Sx_1),axis=0)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
    

        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=False, max_order=1,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.zero_order_feature(degree_vector_A)
        Sx_1 = utils.first_order_feature(np.matmul(W, degree_vector_A))
        Sx_0 = np.concatenate((Sx_0,Sx_1),axis=0)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
 
 
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_second_order_moments(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=False, max_order=2,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.zero_order_feature(degree_vector_A)
        u = np.matmul(W, degree_vector_A)
        Sx_1 = utils.first_order_feature(u)
        Sx_2 = utils.selected_second_order_feature(W, u)
        Sx_0 = np.concatenate((Sx_0,Sx_1, Sx_2),axis=0)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
    

        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=False, max_order=1,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.zero_order_feature(degree_vector_A)
        Sx_1 = utils.first_order_feature(np.matmul(W, degree_vector_A))
        Sx_0 = np.concatenate((Sx_0,Sx_1),axis=0)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
 


class TestNormalizedOrderMoments:
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_normalized_zero_order_moments(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=True, max_order=0,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.normalized_zero_order_feature(degree_vector_A)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
    

        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=True, max_order=0,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.normalized_zero_order_feature(degree_vector_A)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
 
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_normalized_first_order_moments(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=True, max_order=1,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.normalized_zero_order_feature(degree_vector_A)
        u = np.abs(np.matmul(W, degree_vector_A))
        u[np.abs(u) < 1e-12] = 0.0
        Sx_1 = utils.normalized_first_order_feature(u)
        Sx_0 = np.concatenate((Sx_0,Sx_1),axis=0)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
    

        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=True, max_order=1,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.normalized_zero_order_feature(degree_vector_A)
        u = np.abs(np.matmul(W, degree_vector_A))
        u[np.abs(u) < 1e-12] = 0.0
        Sx_1 = utils.normalized_first_order_feature(u)
        Sx_0 = np.concatenate((Sx_0,Sx_1),axis=0)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
 
 
    @pytest.mark.parametrize('backends_devices', backends_devices)
    def test_normalized_second_order_moments(self, backends_devices):
        backend, device = backends_devices
        
        # two clique adjacency matrix
        A = np.array([[0, 1], [1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=True, max_order=2,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.normalized_zero_order_feature(degree_vector_A)
        u = np.abs(np.matmul(W, degree_vector_A))
        u[np.abs(u) < 1e-12] = 0.0
        Sx_1 = utils.normalized_first_order_feature(u)
        Sx_2 = utils.normalized_selected_second_order_feature(W, u)
        Sx_0 = np.concatenate((Sx_0,Sx_1, Sx_2),axis=0)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
    

        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        degree_vector_A = compute_degree_vector(A)
        S = ScatteringGraph(J=5, Q=4, A=A, normalize=True, max_order=2,
                backend=backend).to(device)
        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()
        S_x0 = S(x).cpu().detach().squeeze(-1).numpy()
        W = utils.graph_wavelet(utils.lazy_random_walk(A))
        Sx_0 = utils.normalized_zero_order_feature(degree_vector_A)
        u = np.abs(np.matmul(W, degree_vector_A))
        u[np.abs(u) < 1e-12] = 0.0
        Sx_1 = utils.normalized_first_order_feature(u)
        Sx_2 = utils.normalized_selected_second_order_feature(W, u)
        Sx_0 = np.concatenate((Sx_0,Sx_1, Sx_2),axis=0)
        assert S_x0.shape == Sx_0.shape
        assert np.allclose(S_x0, Sx_0)
 

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
        mean = torch.from_numpy(mean).double().to(device)

        mean_moment = moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[0]])
        var = torch.from_numpy(var).double().to(device)

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
        mean = torch.from_numpy(mean).double().to(device)

        mean_moment = moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[0]])
        var = torch.from_numpy(var).double().to(device)

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
        mean = torch.from_numpy(mean).double().to(device)

        mean_moment = moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[2]])
        var = torch.from_numpy(var).double().to(device)

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
        mean = torch.from_numpy(mean).double().to(device)

        mean_moment = moment(x, 1)
        assert torch.allclose(mean, mean_moment)

        #q = 2
        var = np.array([[12]])
        var = torch.from_numpy(var).double().to(device)

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

        S = ScatteringGraph(J=2, Q=2, A=A, normalize=False, max_order=2)
        S = S.to(device)

        x = compute_degree_vector(A)
        x = torch.from_numpy(x).to(device).double()

        S_x = S(x)
        print(S_x.shape)
        print(S_x.squeeze())
        print("THIS TEST IS MEANT TO FAIL. PLEASE VERIFY THE OUTPUTS ARE CORRECT" )
        assert False



        

