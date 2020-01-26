import numpy as np
import warnings

BACKEND_NAME = 'numpy'
from collections import namedtuple
from scipy.fftpack import fftn, ifftn


def _iscomplex(x):
    return x.dtype == np.complex64 or x.dtype == np.complex128


def complex_modulus(input_array):
    """Computes complex modulus.

        Parameters
        ----------
        input_array : tensor
            Input tensor whose complex modulus is to be calculated.
        Returns
        -------
        modulus : tensor
            Tensor the same size as input_array. modulus[..., 0] holds the
            result of the complex modulus, modulus[..., 1] = 0.

    """

    return np.abs(input_array)


def modulus_rotation(x, module=None):
    """Used for computing rotation invariant scattering transform coefficents.

        Parameters
        ----------
        x : tensor
            Size (batchsize, M, N, O, 2).
        module : tensor
            Tensor that holds the overall sum. If none, initializes the tensor
            to zero (default).
        Returns
        -------
        output : torch tensor
            Tensor of the same size as input_array. It holds the output of
            the operation::
            $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$
            which is covariant to 3D translations and rotations.
    """
    if module is None:
        module = np.zeros_like(x)
    else:
        module = module ** 2
    module += np.abs(x) ** 2
    return np.sqrt(module)


def _compute_standard_scattering_coefs(input_array, filter_list, J, subsample):
    """Computes convolution and downsamples.

        Computes the convolution of input_array with a lowpass filter phi_J
        and downsamples by a factor J.

        Parameters
        ----------
        input_array : torch Tensor
            Size (batchsize, M, N, O, 2).
        filter_list : list of torch Tensors
            Size (M, N, O, 2).
        J : int
            Low pass scale of phi_J.
        subsample : function
            Subsampling function.

        Returns
        -------
        output : tensor
            The result of input_array \\star phi_J downsampled by a factor J.

    """
    low_pass = filter_list[J]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    return subsample(convolved_input, J)


def _compute_local_scattering_coefs(input_array, filter_list, j, points):
    """Compute convolution and returns particular points.

        Computes the convolution of input_array with a lowpass filter phi_j+1
        and returns the value of the output at particular points.
        Parameters
        ----------
        input_array : torch tensor
            Size (batchsize, M, N, O, 2).
        filter_list : list of torch Tensors
            Size (M, N, O, 2).
        j : int
            The lowpass scale j of phi_j.
        points : torch tensor
            Size (batchsize, number of points, 3).
        Returns
        -------
        output : torch tensor
            Torch tensor of size (batchsize, number of points, 1) with the values
            of the lowpass filtered moduli at the points given.
    """
    local_coefs = torch.zeros(input_array.shape[0], points.shape[1], 1)
    low_pass = filter_list[j + 1]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    for i in range(input_array.shape[0]):
        for j in range(points[i].shape[0]):
            x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
            local_coefs[i, j, 0] = convolved_input[
                i, int(x), int(y), int(z), 0]
    return local_coefs


def subsample(input_array, j):
    """Downsamples.
        Parameters
        ----------
        input_array : tensor
            Input tensor of shape (batch, channel, M, N, O, 2).
        j : int
            Downsampling factor.
        Returns
        -------
        out : tensor
            Downsampled tensor of shape (batch, channel, M // 2 ** j, N // 2
            ** j, O // 2 ** j, 2).
    """
    return input_array[..., ::2 ** j, ::2 ** j, ::2 ** j, :].contiguous()


def compute_integrals(input_array, integral_powers):
    """Computes integrals.

        Computes integrals of the input_array to the given powers.
        Parameters
        ----------
        input_array : torch tensor
            Size (B, M, N, O), where B is batch_size, and M, N, O are spatial
            dims.
        integral_powers : list
            List of P positive floats containing the p values used to
            compute the integrals of the input_array to the power p (l_p
            norms).
        Returns
        -------
        integrals : torch tensor
            Tensor of size (B, P) containing the integrals of the input_array
            to the powers p (l_p norms).
    """
    integrals = np.zeros((input_array.shape[0], len(integral_powers)),dtype=np.complex64)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = (input_array ** q).reshape((input_array.shape[0], -1)).sum(axis=1)
    return integrals



def fft(x, direction='C2C', inverse=False):
    """
        Interface with torch FFT routines for 2D signals.

        Example
        -------
        x = torch.randn(128, 32, 32, 2)
        x_fft = fft(x, inverse=True)

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        direction : string
            'C2R' for complex to real, 'C2C' for complex to complex
        inverse : bool
            True for computing the inverse FFT.
            NB : if direction is equal to 'C2R', then an error is raised.

    """
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT.')

    if direction == 'C2R':
        output = np.real(ifftn(x, axes=(-3, -2, -1)))
    elif direction == 'C2C':
        if inverse:
            output = ifftn(x, axes=(-3, -2, -1))
        else:
            output = fftn(x, axes=(-3, -2, -1))

    return output


def cdgmm3d(A, B, inplace=False):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : torch tensor
            Complex torch tensor.
        B : torch tensor
            Complex of the same size as A.
        inplace : boolean, optional
            If set True, all the operations are performed inplace.

        Raises
        ------
        RuntimeError
            In the event that the tensors are not compatibile for multiplication
            (i.e. the final four dimensions of A do not match with the dimensions
            of B), or in the event that B is not complex, or in the event that the
            type of A and B are not the same.
        TypeError
            In the event that x is not complex i.e. does not have a final dimension
            of 2, or in the event that both tensors are not on the same device.

        Returns
        -------
        output : torch tensor
            Torch tensor of the same size as A containing the result of the
            elementwise complex multiplication of A with B.
    """

    if A.shape[-3:] != B.shape[-3:]:
        raise RuntimeError('The tensors are not compatible for multiplication.')

    if not _iscomplex(A) or not _iscomplex(B):
        raise TypeError('The input, filter and output should be complex.')

    if B.ndim != 3:
        raise RuntimeError('The second tensor must be simply a complex array.')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type.')

     
    if inplace:
        return np.multiply(A, B, out=A)
    else:
        return A * B

   
def concatenate(arrays, L):
    S = np.stack(arrays, axis=1)
    S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
    return S


backend = namedtuple('backend',
                     ['name',
                      'cdgmm3d',
                      'fft',
                      'modulus',
                      'modulus_rotation',
                      'subsample',
                      'compute_integrals',
                      'concatenate'])

backend.name = 'numpy'
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.concatenate = concatenate
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation
backend.subsample = subsample
backend.compute_integrals = compute_integrals
backend._compute_standard_scattering_coefs = _compute_standard_scattering_coefs
backend._compute_local_scattering_coefs = _compute_local_scattering_coefs
