import numpy as np

class Resampler():
    def __init__(self, old_samples, old_weights):
        self.weights = old_weights
        self.samples = old_samples
        self.n_samples = len(old_weights)

    def _find_cdf(self):
        cdf = [0]
        for w in self.weights:
            cdf.append(cdf[-1]+w)
        del cdf[0]
        return cdf

    def stratified_ncopies(self):
        # starting seed u
        u = np.random.uniform(0,1/self.n_samples)
        cdf = self._find_cdf(self.weights)

        ncopies=np.zeros(self.n_samples)

        for _ in range(self.n_samples):
            j = 0
            while cdf[j]<u:
                j += 1
            ncopies[j] += 1
            u += 1/self.n_samples
        return ncopies

    def stratified(self):
        ncopies = self.stratified_ncopies()
        new_weights = [1/self.n_copies]
        new_samples = []
        for i in ncopies:
            for j in range(i):
                new_samples.append(self.old_samples[i])
        print(new_samples)
        return new_samples, new_weights