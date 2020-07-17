BACKEND_NAME = 'numpy'

from ...backend.numpy_backend import NumpyBackend, complex_check, real_check


class NumpyBackend1D(NumpyBackend):
    def __init__(self, np):
        super(NumpyBackend1D, self).__init__(np)

    def subsample_fourier(self, x, k):
        """Subsampling in the Fourier domain
        Subsampling in the temporal domain amounts to periodization in the Fourier
        domain, so the input is periodized according to the subsampling factor.
        Parameters
        ----------
        x : tensor
            Input tensor with at least 3 dimensions, where the next to last
            corresponds to the frequency index in the standard PyTorch FFT
            ordering. The length of this dimension should be a power of 2 to
            avoid errors. The last dimension should represent the real and
            imaginary parts of the Fourier transform.
        k : int
            The subsampling factor.
        Returns
        -------
        res : tensor
            The input tensor periodized along the next to last axis to yield a
            tensor of size x.shape[-2] // k along that dimension.
        """
        complex_check(x)
    
        y = x.reshape(-1, k, x.shape[-1] // k)
    
        res = y.mean(axis=(-2,))

        return res
    
    
    def pad(self, x, pad_left, pad_right):
        """Pad real 1D tensors
        1D implementation of the padding function for real PyTorch tensors.
        Parameters
        ----------
        x : tensor
            Three-dimensional input tensor with the third axis being the one to
            be padded.
        pad_left : int
            Amount to add on the left of the tensor (at the beginning of the
            temporal axis).
        pad_right : int
            amount to add on the right of the tensor (at the end of the temporal
            axis).
        Returns
        -------
        output : tensor
            The tensor passed along the third dimension.
        """
        if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
            raise ValueError('Indefinite padding size (larger than tensor).')
        
        paddings = ((0, 0),) * len(x.shape[:-1])
        paddings += (pad_left, pad_right), 
    
        output = self.np.pad(x, paddings, mode='reflect')

        return output
    

    def unpad(self, x, i0, i1):
        """Unpad real 1D tensor
        Slices the input tensor at indices between i0 and i1 along the last axis.
        Parameters
        ----------
        x : tensor
            Input tensor with least one axis.
        i0 : int
            Start of original signal before padding.
        i1 : int
            End of original signal before padding.
        Returns
        -------
        x_unpadded : tensor
            The tensor x[..., i0:i1].
        """
        return x[..., i0:i1]

    
    def concatenate(self, arrays):
        return self.np.stack(arrays, axis=-2)


    def rfft(self, x):
        real_check(x)

        return self.np.fft.fft(x)
    
    
    def irfft(self, x):
        complex_check(x)

        return self.np.fft.ifft(x).real
    
    
    def ifft(self, x):
        complex_check(x)

        return self.np.fft.ifft(x)


class FFTBackend1D(NumpyBackend1D):
    def __init__(self, np, fft):
        super(NumpyBackend1D, self).__init__(np)
        self.np = np
        self.fft = fft
    
    def rfft(self, x):
        real_check(x)

        return self.fft.fft(x)
    
    
    def irfft(self, x):
        complex_check(x)

        return self.fft.ifft(x).real
    

    def ifft(self, x):
        complex_check(x)

        return self.fft.ifft(x)

import numpy
import scipy.fftpack

backend = FFTBackend1D(numpy, scipy.fftpack)
