import numpy as np
import Prob_Utils

class Proposal():
    def __init__(self, old_samples):
        self.old_samples = np.array(old_samples)
        self.new_samples = None
        self.probs = []
    """
    each proposal funciton (child classes) must have an update 
    function which takes the a set of samples and runs the update
    procedure and updates the probability variables
    """
    @abs
    def update(self):
        print(f'This function should have been overriden by child class.')
        return



class Random_Walk(Proposal):
    def __init__(self, step_sizes : list[float], 
                 old_samples : list[list[float]] = None):
        """
        random walk just adds gaussian noise to the current
        proposal, step_sizes should be what you would put on the main
        diag of a covar matrix - it should be a 1d vector
        """
        super().__init__(old_samples)
        self.step_sizes = step_sizes
        self.covar_mat = np.diag(step_sizes)
        if old_samples is not None:
            self.update(old_samples)
        
    def _walk(self):
        noise = np.random.standard_normal(self.old_samples.shape)
        scaled_noise = np.multiply(self.step_sizes , noise)
        self.new_samples = self.old_samples + scaled_noise
    
    def _calc_p(self):
        print(self.covar_mat)
        for old_s, new_s in zip(self.old_samples, self.new_samples):
            p = Prob_Utils.multivariate_normal_p(new_s, old_s, self.covar_mat)
            self.probs.append(p)

    def update(self, samples : list[list[float]]):
        """
        updates the object with the new samples and runs the random
        walk on the new samples
        """
        self.old_samples = samples
        self._walk()
        self._calc_p()
        self.new_samples = np.array(self.new_samples)
        self.probs = np.array(self.probs)


def main():
    n = 5
    old_samples = np.random.rand(n,2)
    step_sizes = [0.2,0.2]
    rw = Random_Walk(step_sizes)
    print(rw.probs)


if __name__ == '__main__':
    main()