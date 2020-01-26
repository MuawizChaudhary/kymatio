""" This script will test the submodules used by the scattering module"""
import os
import io
import numpy as np
import pytest
from kymatio import HarmonicScattering3D

backends = []

from kymatio.scattering3d.backend.numpy_backend import backend
backends.append(backend)


def relative_difference(a, b):
    return np.sum(np.abs(a - b)) / max(np.sum(np.abs(a)), np.sum(np.abs(b)))


@pytest.mark.parametrize("backend", backends)
def test_against_standard_computations(backend):

    file_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_path, 'test_data_3d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)
    x = data['x']
    scattering_ref = data['Sx']
    J = data['J']
    L = data['L']
    integral_powers = data['integral_powers']

    M = x.shape[1]

    batch_size = x.shape[0]

    N, O = M, M
    sigma = 1

    scattering = HarmonicScattering3D(J=J, shape=(M, N, O), L=L,
                                      sigma_0=sigma, method='integral',
                                      integral_powers=integral_powers,
                                      max_order=2, backend=backend,
                                      frontend='numpy')

    order_0 = backend.compute_integrals(x, integral_powers)
    scattering.max_order = 2
    scattering.method = 'integral'
    scattering.integral_powers = integral_powers
    orders_1_and_2 = scattering(x)
    #Jprint(orders_1_and_2)

    # WARNING: These are hard-coded values for the setting J = 2.
    n_order_1 = 3
    n_order_2 = 3

    # Extract orders and make order axis the slowest in accordance with
    # the stored reference scattering transform.
    order_1 = orders_1_and_2[:, 0:n_order_1, ...]
    order_2 = orders_1_and_2[:, n_order_1:n_order_1 + n_order_2, ...]

    # Permute the axes since reference has (batch index, integral power, j,
    # ell) while the computed transform has (batch index, j, ell, integral
    # power).
    order_1 = order_1.transpose((0, 3, 1, 2))
    order_2 = order_2.transpose((0, 3, 1, 2))

    order_1 = order_1.reshape((batch_size, -1))
    order_2 = order_2.reshape((batch_size, -1))

    orders_1_and_2 = np.concatenate((order_1, order_2), 1)

    order_0 = order_0.reshape((batch_size, -1))
    start = 0
    end = order_0.shape[1]
    order_0_ref = scattering_ref[:, start:end]

    orders_1_and_2 = orders_1_and_2.reshape((batch_size, -1))
    start = end
    end += orders_1_and_2.shape[1]
    orders_1_and_2_ref = scattering_ref[:, start:end]

    order_0_diff_cpu = relative_difference(order_0_ref, order_0)
    print(orders_1_and_2_ref.shape)
    print(orders_1_and_2.shape)
    orders_1_and_2_diff_cpu = relative_difference(
        orders_1_and_2_ref, orders_1_and_2)

    assert order_0_diff_cpu < 1e-6, "CPU : order 0 do not match, diff={}".format(order_0_diff_cpu)
    assert orders_1_and_2_diff_cpu < 1e-6, "CPU : orders 1 and 2 do not match, diff={}".format(orders_1_and_2_diff_cpu)

