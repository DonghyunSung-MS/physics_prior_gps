import abc

import numpy as np


class Dynamics(object):
    """Dynamics Moudle superclass"""

    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        """Linearized Dynamics Module

        x_{t+1} = AB_t @ [x_t; u_t] + c_t

        ex) x_{t+1} = f(x_t, u_t) \approx
        x_{t+1} = f(x_t^*, u_t^*) + \nabla f_{xu_t} ([x_t; u_t] - [x_t^*; u_t^*])

        AB_t = \nabla f_{xu_t}
        c_t = f(x_t^*, u_t^*) - nabla f_{xu_t} [x_t^*; u_t^*]

        Args:
            hyperparams ([type]): [description]
        """
        self._hyperparams = hyperparams

        self.AB_t = np.array(np.nan)
        self.c_t = np.array(np.nan)
        self.W_t = np.array(np.nan)

    @abc.abstractclassmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("fit method is not implemented yet!")
