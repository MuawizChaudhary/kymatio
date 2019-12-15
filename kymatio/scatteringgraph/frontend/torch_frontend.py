import numpy as np
import torch
from kymatio.scatteringgraph.core.scatteringgraph import scatteringgraph
from .base_frontend import ScatteringBaseGraph
from ...frontend.torch_frontend import ScatteringTorch


class ScatteringTorchGraph(ScatteringTorch, ScatteringBaseGraph):
    def __init__(self, J, Q=2, A=None, normalize=True, max_order=2, backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBaseGraph.__init__(self, J, Q, A, normalize, max_order, backend)
        ScatteringBaseGraph._instantiate_backend(self, 'kymatio.scatteringgraph.backend.')
        ScatteringBaseGraph.build(self)
        ScatteringBaseGraph.create_filters(self)
        self.register_filters()

    def register_single_filter(self, v, n):
        current_filter = torch.from_numpy(v)
        self.register_buffer('tensor' + str(n), current_filter)
        return current_filter

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters

        n = 0

        for psi in self.psi:
            self.psi[n] = self.register_single_filter(psi, n)
            n = n + 1

    def load_single_filter(self, n, buffer_dict):
        return buffer_dict['tensor' + str(n)]

    def load_filters(self):
        """ This function loads filters from the module's buffers """
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        buffer_dict = dict(self.named_buffers())

        n = 0

        psis = self.psi
        for psi in psis:
            psis[n] = self.load_single_filter(n, buffer_dict)
            n = n + 1
        
        return psis

    def scattering(self, input):
        """ This function computes the functional scattering """
        psi = self.load_filters()
        S = scatteringgraph(input, self.J, self.Q, psi, self.normalize, self.max_order, self.backend)
        return S


__all__ = ['ScatteringTorchGraph']       
