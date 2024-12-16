import numpy as np

import os, sys
sys.path.append(os.getcwd())
from SMC import prob_utils


class Initial_Proposal():
    def __init__(self, n_samples : int = None):
        self.n_samples = n_samples
        self.probs = []
        self.samples = None

    def sample(self):
        """
        sample function returns the initial samples for the SMC
        """
        print(f'This is a base class please choose an initial proposal scheme')
        return None

class Strandard_Gauss_Noise(Initial_Proposal):
    def __init__(self, dim_samples : int , n_samples : int = None):
        super().__init__(n_samples)
        self.dim_samples = dim_samples
    
    def _calc_p(self):
        self.probs = []
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


class Gauss(Initial_Proposal):
    """
    this prior might cause issues could deal with invalid samples
    (so -ve) in the gp part by just giving weight 0 but then might
    cause divide by zero problems if there isn't a resample in the
    next step - i suppose i could use very small p vals but that would
    dispropostionally make the next proposal look good if it went positive
    also its best not to just kill samples this way or 
    best to try not to
    """
    def __init__(self, means : list[float] = [1, 3, 3], 
                 scale : list[float] = [1, 3, 3], 
                 positive_only : bool = True,
                 n_samples : int = None):
        super().__init__(n_samples)
        self.means = means
        self.scale = scale
        self.positive_only = True
        self.dim_samples = len(means)
    
    def _calc_p(self):
        """
        this will be slow the way i am doing it
        """
        self.probs = []
        dirs = np.array(np.meshgrid(*[[-1, 1]] * self.dim_samples)).T.reshape(-1, self.dim_samples)
        for s in self.samples:
            p = 0
            covar = np.diag(self.scale)
            if self.positive_only:
                for d in dirs:
                    p += prob_utils.multivariate_normal_p(s*d, self.means, covar)
            else:
                p = prob_utils.multivariate_normal(s, self.means, covar)
            self.probs.append(p)
        self.probs = np.array(self.probs)
    
    def sample(self)->tuple[np.ndarray, np.ndarray]:
        noise = np.random.standard_normal((self.n_samples, self.dim_samples))
        self.samples = abs(noise*self.scale + self.means)
        self._calc_p()

        return self.probs, np.array(self.samples)

def main():
    import time
    n = 1000
    initial = Gauss(n_samples=n)
    start = time.time()
    p, s = initial.sample()
    end = time.time()
    print(f'{p}\n{s}\n{n} samples were generated and their respective probability was caculated in {end-start} seconds')
    return

if __name__ == '__main__':
    main()