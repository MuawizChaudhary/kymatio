import numpy as np
import tensorflow as tf


class Modulus():
    """This class implements a modulus transform for complex numbers.

    Parameters
    ----------
    x: input complex tensor.

    Returns
    ----------
    output: a complex tensor equal to the modulus of x.

    Usage
    ----------
    modulus = Modulus()
    x_mod = modulus(x)
    """
    def __call__(self, x):
        norm = tf.abs(x)

        return norm

class TensorFlowBackend:
    def __init__(self):
        self.name = 'tensorflow'
        self.modulus = Modulus()

    def complex_check(self, x):
        if not self._is_complex(x):
            raise TypeError('The input should be complex.')
    
    def real_check(self, x):
        if not self._is_real(x):
            raise TypeError('The input should be real.')
    
    def _is_complex(self, x):
        return (x.dtype == np.complex64) or (x.dtype == np.complex128)
    
    def _is_real(self, x):
        return (x.dtype == np.float32) or (x.dtype == np.float64)
  
    def concat(self, arrays, dim):
        return tf.stack(arrays, axis=dim)
    
    def cdgmm(self, A, B):
        """
            Complex pointwise multiplication between (batched) tensor A and tensor B.
            Parameters
            ----------
            A : tensor
                A is a complex tensor of size (B, C, M, N, 2)
            B : tensor
                B is a complex tensor of size (M, N) or real tensor of (M, N)
            inplace : boolean, optional
                if set to True, all the operations are performed inplace
            Returns
            -------
            C : tensor
                output tensor of size (B, C, M, N, 2) such that:
                C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :]
        """
        if not self._is_complex(A):
            raise TypeError('The first input must be complex.')
    
        if A.shape[-len(B.shape):] != B.shape[:]:
            raise RuntimeError('The inputs are not compatible for multiplication.')
    
        if not self._is_complex(B) and not self._is_real(B):
            raise TypeError('The second input must be complex or real.')
    
        return A * B
