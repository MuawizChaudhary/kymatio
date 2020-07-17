import numpy


def input_checks(x):
    if x is None:
        raise TypeError('The input should be not empty.')


def complex_check(x):
    if not _is_complex(x):
        raise TypeError('The input should be complex.')


def real_check(x):
    if not _is_real(x):
        raise TypeError('The input should be real.')


def _is_complex(x):
    return (x.dtype == numpy.complex64) or (x.dtype == numpy.complex128)


def _is_real(x):
    return (x.dtype == numpy.float32) or (x.dtype == numpy.float64)

    
class NumpyBackend():
    def __init__(self, np):
        self.np = np
        self.name = 'numpy'
  

    def modulus(self, x):
        """
            This function implements a modulus transform for complex numbers.
    
            Usage
            -----
            x_mod = modulus(x)
    
            Parameters
            ---------
            x: input complex tensor.
    
            Returns
            -------
            output: a real tensor equal to the modulus of x.
    
        """
        return self.np.abs(x)
   
    
    def cdgmm(self, A, B, inplace=False):
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
    
        if not _is_complex(A):
            raise TypeError('The first input must be complex.')
    
        if A.shape[-len(B.shape):] != B.shape[:]:
            raise RuntimeError('The inputs are not compatible for '
                               'multiplication.')
    
        if not _is_complex(B) and not _is_real(B):
            raise TypeError('The second input must be complex or real.')
    
        if inplace:
            return self.np.multiply(A, B, out=A)
        else:
            return A * B
