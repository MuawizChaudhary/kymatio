"""
Authors: Louis Thiry, Georgios Exarchakis and Michael Eickenberg
All rights reserved, 2017.
"""

__all__ = ['solid_harmonic_filter_bank']

import numpy as np
from scipy.special import sph_harm, factorial
from .utils import get_3d_angles, double_factorial, sqrt


def solid_harmonic_filter_bank(M, N, O, J, L, sigma_0, fourier=True):
    """
        Computes a set of 3D Solid Harmonic Wavelets of scales j = [0, ..., J]
        and first orders l = [0, ..., L].

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        J : int
            maximal scale of the wavelets
        L : int
            maximal first order of the wavelets
        sigma_0 : float
            width parameter of mother solid harmonic wavelet
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        filters : list of ndarray
            the element number l of the list is a torch array array of size
            (J+1, 2l+1, M, N, O, 2) containing the (J+1)x(2l+1) wavelets of order l.
    """
    filters = []
    for l in range(L + 1):
        filters_l = np.zeros((J + 1, 2 * l + 1, M, N, O, 2), dtype='float32')
        for j in range(J+1):
            sigma = sigma_0 * 2 ** j
            solid_harm = solid_harmonic_3d(M, N, O, sigma, l, fourier=fourier)
            filters_l[j, :, :, :, :, 0] = solid_harm.real
            filters_l[j, :, :, :, :, 1] = solid_harm.imag
        filters.append(filters_l)
    return filters


def gaussian_filter_bank(M, N, O, J, sigma_0, fourier=True):
    """
        Computes a set of 3D Gaussian filters of scales j = [0, ..., J].

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        J : int
            maximal scale of the wavelets
        sigma_0 : float
            width parameter of father Gaussian filter
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        gaussians : ndarray
            torch array array of size (J+1, M, N, O, 2) containing the (J+1)
            Gaussian filters.
    """
    gaussians = np.zeros((J + 1, M, N, O, 2))
    for j in range(J + 1):
        sigma = sigma_0 * 2 ** j
        gaussian = gaussian_3d(M, N, O, sigma, fourier=fourier)
        gaussians[j, :, :, :, 0] = gaussian
    return gaussians


def gaussian_3d(M, N, O, sigma, fourier=True):
    """
        Computes a 3D Gaussian filter.

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        sigma : float
            gaussian width parameter
        fourier : boolean
            if true, the Gaussian if computed in Fourier space
	    if false, the Gaussian if computed in signal space

        Returns
        -------
        gaussian : ndarray
            numpy array of size (M, N, O) and type float32 ifftshifted such
            that the origin is at the point [0, 0, 0]
    """
    grid = np.fft.ifftshift(
        np.mgrid[-M // 2:-M // 2 + M,
                 -N // 2:-N // 2 + N,
                 -O // 2:-O // 2 + O].astype('float32'),
        axes=(1,2,3))
    _sigma = sigma
    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        _sigma = 1. / sigma

    gaussian = np.exp(-0.5 * (grid ** 2).sum(0) / _sigma ** 2)
    if not fourier:
        gaussian /= (2 * np.pi) ** 1.5 * _sigma ** 3

    return gaussian


def solid_harmonic_3d(M, N, O, sigma, l, fourier=True):
    """
        Computes a set of 3D Solid Harmonic Wavelets.
	A solid harmonic wavelet has two integer orders l >= 0 and -l <= m <= l
	In spherical coordinates (r, theta, phi), a solid harmonic wavelet is
	the product of a polynomial Gaussian r^l exp(-0.5 r^2 / sigma^2)
	with a spherical harmonic function Y_{l,m} (theta, phi).

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        sigma : float
            width parameter of the solid harmonic wavelets
        l : int
            first integer order of the wavelets
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        solid_harm : ndarray, type complex64
            numpy array of size (2l+1, M, N, 0) and type complex64 containing
            the 2l+1 wavelets of order (l , m) with -l <= m <= l.
            It is ifftshifted such that the origin is at the point [., 0, 0, 0]
    """
    solid_harm = np.zeros((2*l+1, M, N, O), np.complex64)
    grid = np.fft.ifftshift(
        np.mgrid[-M // 2:-M // 2 + M,
                 -N // 2:-N // 2 + N,
                 -O // 2:-O // 2 + O].astype('float32'),
        axes=(1,2,3))
    _sigma = sigma

    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        _sigma = 1. / sigma

    r_square = (grid ** 2).sum(0)
    r_power_l = sqrt(r_square ** l)
    gaussian = np.exp(-0.5 * r_square / _sigma ** 2).astype('complex64')

    if l == 0:
        if fourier:
            return gaussian.reshape((1, M, N, O))
        return gaussian.reshape((1, M, N, O)) / (
                                          (2 * np.pi) ** 1.5 * _sigma ** 3)

    polynomial_gaussian = r_power_l * gaussian / _sigma ** l

    polar, azimuthal = get_3d_angles(grid)

    for i_m, m in enumerate(range(-l, l + 1)):
        solid_harm[i_m] = sph_harm(m, l, azimuthal, polar) * polynomial_gaussian

    if l % 2 == 0:
        norm_factor = 1. / (2 * np.pi * np.sqrt(l + 0.5) * 
                                            double_factorial(l + 1))
    else :
        norm_factor = 1. / (2 ** (0.5 * ( l + 3)) * 
                            np.sqrt(np.pi * (2 * l + 1)) * 
                            factorial((l + 1) / 2))

    if fourier:
        norm_factor *= (2 * np.pi) ** 1.5 * (-1j) ** l
    else:
        norm_factor /= _sigma ** 3

    solid_harm *= norm_factor

    return solid_harm


def gabor_nd(grid_or_shape, orientation, scale, xi0=3 * np.pi / 4, sigma0=.5,
            slant=.5, remove_dc=True, ifftshift=True):
    """Computes one n-dimensional Gabor wavelets given orientation and scale.

    Parameters
    ==========
    grid_or_shape, ndarray-like
        either a (short) list of integers providing grid dimensions
        or a grid of shape (ndim, axis1, axis2[, axis3, ..., axis_ndim]).
        If grid shape is specified, then grid step is integer

    orientation: float or array-like
        Specifies the orientation of the main oscillation of the wavelet.
        If desired wavelets are two-dimensional, then a float value is taken to
        specify the 2D angle. A float value can only be used in 2D.
        If orientation is a n-dimensional vector it is taken to mean the wave
        vector direction of the wavelet.

    scale: float, usually integer
        Specifies the octave scale at which the Gabor is to be generated. This
        value is used to modify xi0 and sigma0 to xi0 / 2 ** scale, and
        sigma0 * 2 ** scale.
    
    xi0: float,
        Specifies the center spatial frequency of the fastest-oscillating
        wavelet. (Modified by scale, see gabor_derivative docstring)

    sigma0: float,
        Specifies the width of the Gaussian envelope at the finest scale
        (Modified by scale parameter)

    slant: float, around 1.
        Determines the ratio between Gaussian width in wave direction versus
        all the other directions. Typically set to <= 1. to yield wide edges
        more selective to orientations.

    remove_dc: boolean, default True
        By default, Gabor filters are not zero-sum in the real part, since the
        integral of a Gaussian times a cosine is not 0. If set to True, a
        Gaussian envelope is subtracted from the real part of the Gabor filter
        to obtain exact zero-sum property. The result is also called Morlet
        wavelet.

    ifftshift: boolean, default True
        When set to True, then the 0-frequency is placed in the top front left
        corner of the grid. When set to False, the 0-frequency is placed in the
        middle of the grid, which is more convenient for visualization.


    Returns
    =======
    wavelet, array, dtype complex128
        The shape is the same as that of the grid."""

    
    grid = check_grid(grid_or_shape)
    ndim = grid.shape[0]
    if isinstance(orientation, numbers.Number):
        if ndim != 2:
            raise ValueError("If specifying orientation as a number, then Gabor "
                                "must be 2D, given {}.".format(ndim))
        orientation = np.array((np.cos(orientation), np.sin(orientation)))
    orientation = orientation.ravel() / np.linalg.norm(orientation)

    _, _, VT = np.linalg.svd(orientation[np.newaxis])
    VT[0] = orientation
    transformed_grid = grid.T.dot(VT.T).T
    sigma, xi = sigma0 * 2. ** scale, xi0 / 2. ** scale
    oscillation = np.exp(1j * xi * transformed_grid[0])
    squash_vector = np.array((1. / slant,) + (1,) * (ndim - 1))
    squashed_grid = (transformed_grid.T * squash_vector).T
    gaussian = (np.exp(-.5 * ((squashed_grid / sigma) ** 2).sum(0)) /
            np.sqrt((2 * np.pi) ** ndim * np.prod(sigma / squash_vector)))
    gabor = gaussian * oscillation
    if remove_dc:
        dc = np.real(gabor.sum())
        morlet = gabor - gaussian / gaussian.sum() * dc
        wavelet = morlet
    else:
        wavelet = gabor
    if ifftshift:
        wavelet = np.fft.ifftshift(wavelet)
    return wavelet



