# Authors: Louis Thiry, Georgios Exarchakis
# Scientific Ancestry: Louis Thiry, Georgios Exarchakis, Matthew Hirn, Michael Eickenberg

from ...frontend.numpy_frontend import ScatteringNumPy
from ..core.scattering3d import scattering3d
from .base_frontend import ScatteringBase3D


class HarmonicScatteringNumPy3D(ScatteringNumPy, ScatteringBase3D):
    """3D Solid Harmonic scattering .
    This class implements solid harmonic scattering on an input 3D image.
    For details see https://arxiv.org/abs/1805.00571.
    Instantiates and initializes a 3d solid harmonic scattering object.
    Parameters
    ----------
    J: int
        number of scales
    shape: tuple of int
        shape (M, N, O) of the input signal
    L: int, optional
        Number of l values. Defaults to 3.
    sigma_0: float, optional
        Bandwidth of mother wavelet. Defaults to 1.
    max_order: int, optional
        The maximum order of scattering coefficients to compute. Must be
        either 1 or 2. Defaults to 2.
    rotation_covariant: bool, optional
        if set to True the first order moduli take the form:
        $\\sqrt{\\sum_m (x \\star \\psi_{j,l,m})^2)}$
        if set to False the first order moduli take the form:
        $x \\star \\psi_{j,l,m}$
        The second order moduli change analogously
        Defaut: True
    method: string, optional
        specifies the method for obtaining scattering coefficients
        ("standard","local","integral"). Default: "standard"
    points: array-like, optional
        List of locations in which to sample wavelet moduli. Used when
        method == 'local'
    integral_powers: array-like
        List of exponents to the power of which moduli are raised before
        integration. Used with method == 'standard', method == 'integral'
    """
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='integral', points=None,
                 integral_powers=(0.5, 1., 2.), backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase3D.__init__(self, J, shape, L, sigma_0, max_order,
                                  rotation_covariant, method, points,
                                  integral_powers, backend)

        self.build()

    def build(self):
        ScatteringBase3D._instantiate_backend(self, 'kymatio.scattering3d.backend.')
        ScatteringBase3D.build(self)
        ScatteringBase3D.create_filters(self)

    def scattering(self, input_array):
        methods = ['integral']
        if not self.method in methods:
            raise ValueError('method must be in {}'.format(methods))
        if self.method == 'integral': \
                self.averaging = lambda x, j: self.backend.compute_integrals(self.backend.fft(x, inverse=True),
                                                                             self.integral_powers)

        return scattering3d(input_array, filters=self.filters, rotation_covariant=self.rotation_covariant, L=self.L,
                            J=self.J, max_order=self.max_order, backend=self.backend, averaging=self.averaging)

    def forward(self, input_array):
        """
        The forward pass of 3D solid harmonic scattering
        Parameters
        ----------
        input_array: numpy array
            input of size (batchsize, M, N, O)
        Returns
        -------
        output: tuple | numpy array
            if max_order is 1 it returns a numpy array with the
            first order scattering coefficients
            if max_order is 2 it returns a numpy array with the
            first and second order scattering coefficients,
            concatenated along the feature axis
        """
        if not type(input_array) is np.ndarray:
            raise TypeError('The input should be a NumPy array.')
        
        if input_array.dim() < 3:
            raise RuntimeError('Input tensor must have at least three '
                               'dimensions.')

        if (input_array.shape[-1] != self.O or input_array.shape[-2] != self.N
            or input_array.shape[-3] != self.M):
            raise RuntimeError(
                'Tensor must be of spatial size (%i, %i, %i).' % (
                    self.M, self.N, self.O))

        batch_shape = input_array.shape[:-3]
        signal_shape = input_array.shape[-3:]

        input_array = input_array.reshape((-1,) + signal_shape)

        S = self.scattering(input_array)
        scattering_shape = S.shape[1:]

        S = S.reshape(batch_shape + scattering_shape)

        return S


__all__ = ['HarmonicScatteringNumPy3D']
