import numpy as np
"""
Stratified really should be its own class that just inherits from 
the base class Resampler - but would only really need to change this if 
I added new resampling methods that are actually useful like
conditional importance resampling or something similar although
CIRS would need all the methods that stratified has anyway

so this should be rewritten to be more like Initial_proposals / proposals
"""


class Resampler():
    def __init__(self, old_samples : list[list[float]] = None, 
                 old_weights : list[float] = None):
        self.weights = old_weights
        self.samples = old_samples
    
    def resample(self):
        print('This is a base class please specify a different class')
        return None

class Staratified(Resampler):
    def __init__(self, old_samples : list[list[float]] = None, 
                 old_weights : list[float] = None):
        super().__init__(old_samples, old_weights)


    def _find_cdf(self):
        return np.cumsum(self.weights)

    def _stratified_ncopies(self):
        # starting seed u
        u0 = np.random.uniform(0, 1 / self.n_samples)  # Initial offset
        cdf = self._find_cdf()
        u = u0 + np.arange(self.n_samples) / self.n_samples

        indices = np.zeros(self.n_samples, dtype=int)
        j = 0

        for i in range(self.n_samples):
            while j < len(cdf) and u[i] > cdf[j]:
                j += 1
            indices[i] = j

        return indices

    def resample(self, old_samples : list[list[float]], old_weights : list[float]):
        self.weights = old_weights
        self.samples = old_samples
        self.n_samples = len(old_weights)

        # Compute resampling indices
        indices = self._stratified_ncopies()
        # Resample
        new_samples = self.samples[indices]
        new_weights = np.ones_like(self.weights) / self.n_samples
        return new_samples, new_weights

def main():

    return

if __name__ == '__main__':
    main()