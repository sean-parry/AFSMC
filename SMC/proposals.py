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

    def update(self, samples : list[float]):
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


class Defensive_Sampling(Proposal):
    def __init__(self, step_probs : list[float] = [0.9, 0.1], 
                 step_sizes : list[float] = [0.1, 1.0], 
                 old_samples : list[list[float]] = None):
        super().__init__(old_samples)

        if sum(step_probs) == 1:
            self.step_probs = step_probs
        else:
            print('Warning step_probs should sum to one, step_probs array given has been normalised')
            self.step_probs = step_probs / sum(step_probs)

        self.cdf_probs = [0.0]
        for step_prob in step_probs:
            self.cdf_probs.append(self.cdf_probs[-1] + step_prob)
        self.step_sizes = step_sizes
        if old_samples is not None:
            self.update(old_samples)
        return
    
    def _walk(self):
        # could have probably just used np.choice and made an array of step sizes 
        # then just used some basic matrix maths, change that i guess if its making 
        # it slow
        unif_probs = np.random.rand(self.old_samples.shape[0])
        noise_arr = np.random.standard_normal(self.old_samples.shape)
        self.new_samples = np.zeros(self.old_samples.shape)
        for i, (unif_prob, unscaled_noise, old_sample) in enumerate(zip(unif_probs, noise_arr, self.old_samples)):
            for cdf_prob, step_size in zip(self.cdf_probs[1:], self.step_sizes):
                if unif_prob<cdf_prob:
                    self.new_samples[i] = old_sample + unscaled_noise * step_size
                    break
    
    def _calc_p(self):
        self.probs = np.zeros(self.old_samples.shape[0])
        for i, (old_sample, new_sample) in enumerate(zip(self.old_samples, self.new_samples)):
            for w, step in zip(self.step_probs, self.step_sizes):
                self.probs[i] += w * prob_utils.multivariate_normal_p(new_sample, old_sample, self.identity*step)
        self.probs = np.array(self.probs)
    

    def update(self, samples : list[list[float]]):
        """
        updates the object with the new samples and runs the random
        walk on the new samples
        """
        samples = np.array(samples)
        self.identity = np.eye(samples.shape[1])
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
    df = Defensive_Sampling()
    df.update(np.random.rand(10,2))
    print(df.new_samples)
    print(df.probs)
    return
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
    return


if __name__ == '__main__':
    main()