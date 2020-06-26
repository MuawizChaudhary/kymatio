from ..backend.jax_backend import input_checks

class ScatteringJax:
    def __init__(self):
        self.frontend_name = 'jax'

    def scattering(self, x):
        """ This function should compute the scattering transform."""
        raise NotImplementedError

    def __call__(self, x):
        """This method is an alias for `scattering`."""

        input_checks(x)

        return self.scattering(x)

    _doc_array = 'jax.numpy.ndarray'
    _doc_array_n = 'n'

    _doc_alias_name = '__call__'

    _doc_alias_call = ''

    _doc_frontend_paragraph = ''

    _doc_sample = ''

    _doc_has_shape = True

    _doc_has_out_type = True
