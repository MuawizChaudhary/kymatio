import tensorflow as tf
from packaging import version


assert (version.parse(tf.__version__) >= version.parse('2.0.0a0'),
        'Current TensorFlow version is ' + str(tf.__version__) + '. '
        'Please upgrade TensorFlow to 2.0.0a0 or later.')


class ScatteringTensorFlow(tf.Module):
    def __init__(self, name):
        super(ScatteringTensorFlow, self).__init__(name=name)
        self.frontend_name = 'tensorflow'

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError

    @tf.Module.with_name_scope
    def __call__(self, x):
        """ This function should call the functional scattering."""
        return self.scattering(x)
