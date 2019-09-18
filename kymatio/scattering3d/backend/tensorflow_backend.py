import tensorflow as tf
import torch 
import warnings

BACKEND_NAME = 'torch'
from collections import namedtuple

def iscomplex(input):
    return input.size(-1) == 2

def complex_modulus(input_array):
    modulus = torch.zeros_like(input_array)
    modulus[..., 0] += torch.sqrt((input_array ** 2).sum(-1))
    return modulus



def fft(x, direction='C2C', inverse=False):
    """
        fft of a 3d signal

        Example
        -------
        x = torch.randn(128, 32, 32, 32, 2)
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
.
    """
    if direction == 'C2R':
        if not inverse:
            raise RuntimeError('C2R mode can only be done with an inverse FFT')

    if direction == 'C2R':
        output = tf.real(tf.signal.ifft3d(x, name='irfft3d'))
    elif direction == 'C2C':
        if inverse:
            output = tf.signal.ifft3d(x, name='ifft3d')
        else:
            output = tf.signal.fft3d(x, name='fft3d')
    return output
   

def cdgmm3d(A, B, inplace=False):
    """
    Pointwise multiplication of complex tensors.

    ----------
    A: complex torch tensor
    B: complex torch tensor of the same size as A
    inplace : boolean, optional
        if set True, all the operations are performed inplace
    Returns
    -------
    output : torch tensor of the same size as A containing the result of the 
             elementwise complex multiplication of A with B 
    """
    if B.ndim != 3:
        raise RuntimeError('The dimension of the second input must be 3.')
    return A * B 
   

def finalize(s_order_1, s_order_2, max_order):
    s_order_1 = torch.stack(s_order_1, 2)
    if max_order == 2:
        s_order_2 = torch.stack(s_order_2, 2)
        return torch.cat([s_order_1, s_order_2], dim=1)
    else:
        return s_order_1



def modulus_rotation(x, module):
    """
    Computes the convolution with a set of solid harmonics of scale j and
    degree l and returns the square root of their squared sum over m

    Parameters
    ----------
    input_array : tensor
        size (batchsize, M, N, O, 2)
    l : int
        solid harmonic degree l

    j : int
        solid harmonic scale j

    Returns
    -------

    output : torch tensor
        tensor of the same size as input_array. It holds the output of
        the operation::

        $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$

        which is covariant to 3D translations and rotations

    """
    if module is None:
        module = tf.zeros_like(x)
    else:
        module = module ** 2
    module += x * tf.math.conj(x)
    return tf.sqrt(module)



def _compute_standard_scattering_coefs(input_array, filter, J, subsample):
    """
    Computes the convolution of input_array with a lowpass filter phi_J
    and downsamples by a factor J.

    Parameters
    ----------
    input_array: torch tensor of size (batchsize, M, N, O, 2)
    filter: torch tensor
        size (M, N, O, 2)

    Returns
    -------
    output: the result of input_array \\star phi_J downsampled by a factor J

    """
    low_pass = filter[J]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    return subsample(convolved_input, J)



def _compute_local_scattering_coefs(input_array, filter, j, points):
    """
    Computes the convolution of input_array with a lowpass filter phi_j+1 and
    and returns the value of the output at particular points

    Parameters
    ----------
    input_array: torch tensor
        size (batchsize, M, N, O, 2)
    filter: torch tensor
        size (M, N, O, 2)
    j: int
        the lowpass scale j of phi_j
    points: torch tensor
        size (batchsize, number of points, 3)

    Returns
    -------
    output: torch tensor of size (batchsize, number of points, 1) with
            the values of the lowpass filtered moduli at the points given.

    """
    local_coefs = torch.zeros(input_array.size(0), points.size(1), 1)
    low_pass = filter[j+1]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    for i in range(input_array.size(0)):
        for j in range(points[i].size(0)):
            x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
            local_coefs[i, j, 0] = convolved_input[
                i, int(x), int(y), int(z), 0]
    return local_coefs



def subsample(input_array, j):
    return input_array[..., ::2 ** j, ::2 ** j, ::2 ** j, :].contiguous()



def compute_integrals(input_array, integral_powers):
    """
        Computes integrals of the input_array to the given powers.

        Parameters
        ----------
        input_array: torch tensor
            size (B, M, N, O), B batch_size, M, N, O spatial dims

        integral_powers: list
            list of P positive floats containing the p values used to
            compute the integrals of the input_array to the power p (l_p norms)

        Returns
        -------
        integrals: torch tensor
            tensor of size (B, P) containing the integrals of the input_array
            to the powers p (l_p norms)

    """
    #integrals = tf.zeros((input_array.shape[0], len(integral_powers)),dtype=tf.complex64)
    integrals = []

    for i_q, q in enumerate(integral_powers):
        integrals.append(tf.reduce_sum(tf.reshape((input_array ** \
                q), shape=(input_array.shape[0], -1)), axis=1))

    return tf.expand_dims(tf.stack(integrals, axis=-1), axis=-1)


def aggregate(x):
    return torch.stack([arr[..., 0] for arr in x], 1)

backend = namedtuple('backend', ['name', 'cdgmm3d', 'fft', 'finalize', 'modulus', 'modulus_rotation', 'subsample',\
                                 'compute_integrals', 'aggregate'])

backend.name = 'tensorflow'
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.aggregate = aggregate
backend.finalize = finalize
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation
backend.subsample = subsample
backend.compute_integrals = compute_integrals
backend._compute_standard_scattering_coefs = _compute_standard_scattering_coefs
backend._compute_local_scattering_coefs = _compute_local_scattering_coefs

