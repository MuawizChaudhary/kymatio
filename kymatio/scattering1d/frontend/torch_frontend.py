import torch
import warnings

from ...frontend.torch_frontend import ScatteringTorch
from ..core.scattering1d import scattering1d
from ..utils import precompute_size_scattering
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
            oversampling=0, vectorize=True, out_type='array', backend='torch', 
            complex_input=False):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, vectorize, out_type, backend, complex_input)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n = 0
        # prepare for pytorch
        for k in self.phi_f.keys():
            if type(k) != str:
                # view(-1, 1).repeat(1, 2) because real numbers!
                self.phi_f[k] = torch.from_numpy(
                    self.phi_f[k]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), self.phi_f[k])
                n += 1
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    # view(-1, 1).repeat(1, 2) because real numbers!
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float().view(-1, 1)
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    # view(-1, 1).repeat(1, 2) because real numbers!
                    psi_f[sub_k] = torch.from_numpy(
                        psi_f[sub_k]).float().view(-1, 1)
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1

    def load_filters(self):
        """This function loads filters from the module's buffer """
        buffer_dict = dict(self.named_buffers())
        n = 0

        for k in self.phi_f.keys():
            if type(k) != str:
                self.phi_f[k] = buffer_dict['tensor' + str(n)]
                n += 1

        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

    def scattering(self, x):
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        if not self.average and self.out_type == 'array' and self.vectorize:
            raise ValueError("Options average=False, out_type='array' and "
                             "vectorize=True are mutually incompatible. "
                             "Please set out_type to 'list' or vectorize to "
                             "False.")

        if not self.vectorize:
            warnings.warn("The vectorize option is deprecated and will be "
                          "removed in version 0.3. Please set "
                          "out_type='list' for equivalent functionality.",
                          DeprecationWarning)

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]

        x = x.reshape((-1, 1) + signal_shape)

        self.load_filters()

        # get the arguments before calling the scattering
        # treat the arguments
        if self.vectorize:
            size_scattering = precompute_size_scattering(
                self.J, self.Q, self.T, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0


        S = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, 
                         self.log2_T, self.psi1_f, self.psi2_f, self.phi_f,
                         max_order=self.max_order, average=self.average,
                         pad_left=self.pad_left, pad_right=self.pad_right,
                         ind_start=self.ind_start, ind_end=self.ind_end,
                         oversampling=self.oversampling, 
                         vectorize=self.vectorize,
                         size_scattering=size_scattering,
                         out_type=self.out_type)

        if self.out_type == 'array' and self.vectorize:
            scattering_shape = S.shape[-2:]
            new_shape = batch_shape + scattering_shape

            S = S.reshape(new_shape)
        elif self.out_type == 'array' and not self.vectorize:
            for k, v in S.items():
                # NOTE: Have to get the shape for each one since we may have
                # average == False.
                scattering_shape = v.shape[-2:]
                new_shape = batch_shape + scattering_shape

                S[k] = v.reshape(new_shape)
        elif self.out_type == 'list':
            for x in S:
                scattering_shape = x['coef'].shape[-1:]
                new_shape = batch_shape + scattering_shape

                x['coef'] = x['coef'].reshape(new_shape)

        return S


ScatteringTorch1D._document()


def timefrequency_scattering(x, pad, unpad, backend, J, psi1, psi2, phi, psi_fr, 
                             pad_left=0,
        pad_right=0, ind_start=None, ind_end=None, oversampling=0,
        size_scattering=(0, 0, 0), out_type='array'):
    pass


class TimeFrequencyScatteringTorch(ScatteringTorch1D, TimeFrequencyScatteringBase):
    def __init__(self, J, shape, 
                 Q=1, T=None, F=None, oversampling=0, J_fr=None, Q_fr=1,
                 out_type='array', backend='torch'):
        vectorize = True # for compatibility, will be removed in 0.3

        # Second-order scattering object for the time variable
        max_order_tm = 2
        ScatteringTorch1D.__init__(self, J, shape, 
                                   Q=Q, T=T, max_order=max_order_tm,
                                   oversampling=oversampling, out_type=out_type, 
                                   backend=backend)

        # First-order scattering object for the frequency variable
        shape_fr = (Q * J)
        self.J_fr = self.get_J_fr if not J_fr else J_fr
        self.Q_fr = Q_fr

        self.sc_freq = ScatteringTorch1D(
            self.J_fr, shape_fr, 
            Q=self.Q_fr, T=F, max_order=1, oversampling=oversampling, 
            out_type=out_type, backend=backend, complex_input=True)

    def scattering(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

        if not self.average and self.out_type == 'array':
            raise ValueError("Options average=False and out_type='array'"
                             "are mutually incompatible. "
                             "Please set out_type='list'.")

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)

        # Load filters
        self.load_filters()
        self.sc_freq.load_filters()

        # Precompute output size
        size_scattering = 1 + self.J * (2*self.get_J_fr() + 1)

        S = timefrequency_scattering(
            x,
            self.backend.pad, self.backend.unpad,
            self.backend,
            self.J,
            self.psi1_f, self.psi2_f, self.phi_f,
            self.sc_freq,
            average=self.average,
            pad_left=self.pad_left, pad_right=self.pad_right,
            ind_start=self.ind_start, ind_end=self.ind_end,
            oversampling=self.oversampling,
            size_scattering=size_scattering,
            out_type=self.out_type)

        # TODO switch-case out_type array vs list
        return S

TimeFrequencyScatteringTorch._document()









__all__ = ['ScatteringTorch1D']
