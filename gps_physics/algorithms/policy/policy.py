import abc

import numpy as np


class Policy(object):
    "Policy Module Superclass"
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        """
        Linear Gaussian Policy
        ps. Nerual Net Policy can be linearized via sampling or input gradient

        u_{t} = N(K_t x_t _+ k_t, Q_uu^-1) i.e maximum entropy l.g
        """
        self._hyperparams = hyperparams

        self.K = np.array(np.nan)
        self.k = np.array(np.nan)
        self.cov = np.array(np.nan)

    @abc.abstractclassmethod
    def get_action(self, *args, **kwargs):
        raise NotImplementedError("get_action method is not implemented yet!")

    @abc.abstractclassmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("fit method is not implemented yet!")
