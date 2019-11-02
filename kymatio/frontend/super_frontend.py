        """ This function should run a filterbank function that
        will create the filters as numpy array, and then, it should
        save those arrays as module's buffers. """
        raise NotImplementedError

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError
    
    def loginfo(self):
        """ Returns the logging message when the frontend is deployed."""
        raise NotImplementedError

