import pytest
import torch
import numpy as np

backends = []

skcuda_available = False
try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        skcuda_available = True
except:
    Warning('torch_skcuda backend not available.')

if skcuda_available:
    from kymatio.scattering1d.backend.torch_skcuda_backend import backend
    backends.append(backend)

from kymatio.scattering1d.backend.torch_backend import backend
backends.append(backend)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_pad_1d(device, backend, random_state=42):
    """
    Tests the correctness and differentiability of pad_1d
    """
    torch.manual_seed(random_state)
    N = 128
    for pad_left in range(0, N - 16, 16):
        for pad_right in [pad_left, pad_left + 16]:
            x = torch.randn(2, 4, N, requires_grad=True, device=device)
            x_pad = backend.pad(x, pad_left, pad_right)
            print(x_pad.shape, x_pad.shape[:-1], pad_left, pad_right)

            x_pad = x_pad.reshape(x_pad.shape[:-1])
            # Check the size
            x2 = x.clone()
            x_pad2 = x_pad.clone()
            for t in range(1, pad_left + 1):
                assert torch.allclose(x_pad2[..., pad_left - t],x2[..., t] + 10)
            for t in range(x2.shape[-1]):
                assert torch.allclose(x_pad2[..., pad_left + t], x2[..., t])
            for t in range(1, pad_right + 1):
                assert torch.allclose(x_pad2[..., x_pad.shape[-1] - 1 - pad_right + t], x2[..., x.shape[-1] - 1 - t])
            # check the differentiability
            loss = 0.5 * torch.sum(x_pad**2)
            loss.backward()
            # compute the theoretical gradient for x
            x_grad_original = x.clone()
            x_grad = x_grad_original.new(x_grad_original.shape).fill_(0.)
            x_grad += x_grad_original
            for t in range(1, pad_left + 1):
                x_grad[..., t] += x_grad_original[..., t]
            for t in range(1, pad_right + 1):  # it is counted twice!
                t0 = x.shape[-1] - 1 - t
                x_grad[..., t0] += x_grad_original[..., t0]
            # get the difference
            assert torch.allclose(x.grad, x_grad)
    # Check that the padding shows an error if we try to pad
    with pytest.raises(ValueError):
        backend.pad(x, x.shape[-1], 0)
    with pytest.raises(ValueError):
        backend.pad(x, 0, x.shape[-1])


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_modulus(device, backend, random_state=42):
    """
    Tests the stability and differentiability of modulus
    """
    if backend.name == "torch_skcuda" and device == "cpu":
        with pytest.raises(TypeError) as re:
            x_bad = torch.randn((4, 2)).cpu()
            backend.modulus(x_bad)
        assert "for CPU tensors" in re.value.args[0]
        return

    torch.manual_seed(random_state)
    # Test with a random vector
    x = torch.randn(2, 4, 128, 2, requires_grad=True, device=device)

    x_abs = backend.modulus(x).squeeze(-1)
    assert len(x_abs.shape) == len(x.shape[:-1])

    # check the value
    x_abs2 = x_abs.clone()
    x2 = x.clone()
    assert torch.allclose(x_abs2, torch.sqrt(x2[..., 0] ** 2 + x2[..., 1] ** 2))

    with pytest.raises(TypeError) as te:
        x_bad = torch.randn(4).to(device)
        backend.modulus(x_bad)
    assert "should be complex" in te.value.args[0]

    if backend.name == "torch_skcuda":
        pytest.skip("The skcuda backend does not pass differentiability"
            "tests, but that's ok (for now).")

    # check the gradient
    loss = torch.sum(x_abs)
    loss.backward()
    x_grad = x2 / x_abs2[..., None]
    assert torch.allclose(x.grad, x_grad)


    # Test the differentiation with a vector made of zeros
    x0 = torch.zeros(100, 4, 128, 2, requires_grad=True, device=device)
    x_abs0 = backend.modulus(x0)
    loss0 = torch.sum(x_abs0)
    loss0.backward()
    assert torch.max(torch.abs(x0.grad)) <= 1e-7


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("device", devices)
def test_subsample_fourier(backend, device, random_state=42):
    """
    Tests whether the periodization in Fourier performs a good subsampling
    in time
    """
    if backend.name == 'torch_skcuda' and device == 'cpu':
        with pytest.raises(TypeError) as re:
            x_bad = torch.randn((4, 2)).cpu()
            backend.subsample_fourier(x_bad, 1)
        assert "for CPU tensors" in re.value.args[0]
        return
    rng = np.random.RandomState(random_state)
    J = 10
    x = rng.randn(2, 4, 2**J) + 1j * rng.randn(2, 4, 2**J)
    x_f = np.fft.fft(x, axis=-1)[..., np.newaxis]
    print(x_f.shape, x.shape)
    x_f.dtype = 'float64'  # make it a vector
    x_f_th = torch.from_numpy(x_f).to(device)
    print(x_f.shape, x_f_th.shape)

    for j in range(J + 1):
        x_f_sub_th = backend.subsample_fourier(x_f_th, 2**j).cpu()
        x_f_sub = x_f_sub_th.numpy()
        x_f_sub.dtype = 'complex128'
        x_sub = np.fft.ifft(x_f_sub[..., 0], axis=-1)
        assert np.allclose(x[:, :, ::2**j] + 1, x_sub)

    # If we are using a GPU-only backend, make sure it raises the proper
    # errors for CPU tensors.
    if device=='cuda':
        with pytest.raises(TypeError) as te:
            x_bad = torch.randn(4).cuda()
            backend.subsample_fourier(x_bad, 1)
        assert "should be complex" in te.value.args[0]
