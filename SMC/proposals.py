import numpy as np

import os, sys
sys.path.append(os.getcwd())
from SMC import prob_utils, kernels

class Proposal():
    def __init__(self, old_samples  : list[list[float]] = None):
        self.old_samples = np.array(old_samples)
        self.new_samples = None
        self.probs = []
    """
    each proposal funciton (child classes) must have an update 
    function which takes the a set of samples and runs the update
    procedure and updates the probability variables
    """
    def update(self, old_samples):
        print(f'This is a base class please choose a proposal scheme')
        return


class Random_Walk(Proposal):
    def __init__(self, step_sizes : list[float] = None, 
                 old_samples : list[list[float]] = None):
        """
        random walk just adds gaussian noise to the current
        proposal, step_sizes should be what you would put on the main
        diag of a covar matrix - it should be a 1d vector
        """
        super().__init__(old_samples)
        self.step_sizes = step_sizes
        self.covar_mat = np.diag(step_sizes) if step_sizes is not None else None
        if old_samples is not None:
            self.update(old_samples)
        
    def _walk(self):
        noise = np.random.standard_normal(self.old_samples.shape)
        scaled_noise = self.step_sizes * noise
        self.new_samples = self.old_samples + scaled_noise
    
    def _calc_p(self):
        self.probs = []
        for old_s, new_s in zip(self.old_samples, self.new_samples):
            p = prob_utils.multivariate_normal_p(new_s, old_s, self.covar_mat)
            self.probs.append(p)

    def update(self, samples : list[list[float]]):
        """
        updates the object with the new samples and runs the random
        walk on the new samples
        """
        samples = np.array(samples)

        if self.step_sizes is None:
            print(f'Warning: No \'step_size\' given using I for covar matrix')
            self.step_sizes = [1]*samples.shape[1]
            self.covar_mat = np.diag(self.step_sizes)
        self.old_samples = samples
        self._walk()
        self._calc_p()
        self.new_samples = np.array(self.new_samples)
        self.probs = np.array(self.probs)

class NUTS(Proposal):
    def __init__(self, log_target = None, mass_vec: list[float] = None,
                 old_samples : list[list[float]] = None):
        """
        minimises the return from the target function
        """
        super().__init__(old_samples)
        self.target = log_target
        self.mass_vec = np.array(mass_vec)
        self.mass_matrix = np.diag(self.mass_vec) if not None else None

        if old_samples is not None:
            self.update(old_samples)
    
    def _calc_p(self):
        return
        self.probs = kernels.Approx_Opt(-self.momentum_vecs, self.new_samples).probs

    def update(self, samples: list[list[float]]):
        """
        updates samples based on the gradient of the target function, with the
        aim of minimising the return from the target function, - so if you want
        to maximise a target send the log of the target - if you want the min
        then just send the target
        """
        samples = np.array(samples)
        if self.target is None:
            print(f'must assign a target function to object to run update')
        if self.mass_matrix is None:
            print('Warning, no mass vector given to object, using I for mass matrix')
            self.mass_matrix = np.eye(samples.shape[1])

        self.old_samples = samples
        self.momentum_vecs = np.random.standard_normal(self.old_samples.shape) * self.mass_vec
        

        self._calc_p


def main():
    rw = Random_Walk()
    rw.update(np.random.rand(10,2))
    rw.update(np.random.rand(10,2))
    return
    def target(sample):
        return prob_utils.multivariate_normal_p(sample, [2,2],np.eye(2))
    n = 500
    old_samples = np.random.rand(n,2)
    step_sizes = [0.2,0.2]
    nuts = NUTS(target, [1, 1])
    nuts.update(old_samples)



if __name__ == '__main__':
    main()