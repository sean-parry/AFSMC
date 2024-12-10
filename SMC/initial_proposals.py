import numpy as np

import os, sys
sys.path.append(os.getcwd())
from SMC import prob_utils


class Initial_Proposal():
    def __init__(self, dim_samples : int = None, n_samples : int = None):
        self.n_samples = n_samples
        self.dim_samples = dim_samples
        self.probs = []
        self.samples = None

    def sample(self):
        """
        sample function returns the initial samples for the SMC
        """
        print(f'This is a base class please choose an initial proposal scheme')
        return None

class Strandard_Gauss_Noise(Initial_Proposal):
    def __init__(self,  dim_samples : int , n_samples : int = None):
        super().__init__(dim_samples, n_samples)
    
    def _calc_p(self):
        for s in self.samples:
            p = prob_utils.multivariate_normal_p(s, [0]*self.dim_samples, np.eye(self.dim_samples))
            self.probs.append(p)

    def sample(self):
        """
        sample function returns the initial samples for the SMC
        computes samples of standard gaussian nosie Xi ~ N([0]*dim_samples ,I) for i = 1, ..., n_samples
        and the probablility of generating each sample
        """
        self.samples = np.random.standard_normal((self.n_samples, self.dim_samples))
        self._calc_p()
        return self.probs, self.samples
