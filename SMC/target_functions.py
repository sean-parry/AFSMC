import gpflow
from gpflow.kernels import Matern52 as mt52
import numpy as np
import tensorflow as tf

import os, sys
sys.path.append(os.getcwd())
from SMC import prob_utils


class base_target():
    def p_sample_batch(sample : list[list[float]])->list[float]:
        return
    def update_xy(self, X : list[list[float]] = None, y : list[list[float]] = None):
        print('No params to update in this method')
        return

    
class normal_dist_2d(base_target):
    def p_sample_batch(self, samples : list[list[float]], mean : list[float] = [3,2], scale : float = 0.5):
        probs = []
        for sample in samples:
            probs.append(prob_utils.multivariate_normal_p(sample, mean, np.eye(2)*scale))
        return probs

class gp_fit(base_target):
    """
    sets aside data for a test train split for each p sample evaluates how
    well the params (sample) fits the data (something we want to maximise)
    not sure if this is the best way or even the way that gpflow does it in
    their optimizers
    """
    def __init__(self, X_all : list[list[float]] = None,
                 y_all : list[list[float]] = None,
                 train_test_split_f = 0.8):
        if X_all is not None and y_all is not None:
            self.X_all = np.array(X_all)
            self.y_all = np.array(y_all)
            self.n_samples = self.y_all.shape[0]
        self.train_test_split_f = train_test_split_f
    
    def _train_test_splitter(self, train_test_split : float
                             )-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        could have just used scipy train_test_split, if this is slow switch to that
        although its doubtful this would ever get over 200 training points given its for
        BO of expensive to eval models

        I don't think that the returned probabilities need normalising
        since we normalise the weights in the smc but im not 100%
        """
        test_size = int(np.ceil((1-train_test_split)*self.n_samples))
        test_indexes = np.sort(np.random.choice(self.n_samples, size=test_size, replace=False))

        X_train, X_test , y_train, y_test = [], [], [], []
        pointer = 0
        for i, (x, y) in enumerate(zip(self.X_all, self.y_all)):
            if pointer<test_size and i == test_indexes[pointer]:
                pointer += 1
                X_test.append(x)
                y_test.append(y)
            else:
                X_train.append(x)
                y_train.append(y)
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
        
    def p_sample_batch(self, samples : list[list[float]]):
        if self.X_all is None or self.y_all is None:
            print('error X, or y is not defined')
            return
        
        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_splitter(self.train_test_split_f)
        probs = []
        for sample in samples:
            if all(s>0 for s in sample):
                variance = sample[:1][0]
                lengths = (sample[1:])
                kernel = mt52(variance=variance, lengthscales=lengths)
                model = gpflow.models.GPR((self.X_train, self.y_train),
                                kernel=kernel)
            
                mean, cov = model.predict_y(self.X_test, full_cov=False)
                mean = np.squeeze(mean.numpy())
                cov = np.squeeze(cov.numpy())
                probs.append(prob_utils.multivariate_normal_p(mean, np.squeeze(self.y_test), np.diag(cov)))
            else:
                """
                can't fit a gp with 0 or -ve vals so just assign arbitrary small
                value and warn the user that there is something wrong with their
                proposal
                """
                print(f'Warning: negative or 0 value found in samples')
                probs.append(1e-16)
        return np.log(np.array(probs))
    
    def sample(self, sample):
        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_splitter(self.train_test_split_f)
        if all(s>0 for s in sample):
                variance = sample[:1][0]
                lengths = (sample[1:])
                kernel = mt52(variance=variance, lengthscales=lengths)
                model = gpflow.models.GPR((self.X_train, self.y_train),
                                kernel=kernel)
            
                mean, cov = model.predict_y(self.X_test, full_cov=False)
                mean = np.squeeze(mean.numpy())
                cov = np.squeeze(cov.numpy())
                prob = prob_utils.multivariate_normal_p(mean, np.squeeze(self.y_test), np.diag(cov))
        else:
            """
            can't fit a gp with 0 or -ve vals so just assign arbitrary small
            value and warn the user that there is something wrong with their
            proposal
            """
            print(f'Warning: negative or 0 value found in samples')
            prob = 1e-16
        return prob

    def update_xy(self, X : list[list[float]], y : list[list[float]]):
        self.X_all = np.array(X)
        self.y_all = np.array(y)
        self.n_samples = self.y_all.shape[0]
    



def main():
    sample = [1,-2,1]
    ans = all([s>0 for s in sample])
    print(ans)
    return
    import time
    n = 100
    X_train = np.random.rand(n,2)
    y_train = np.random.rand(n,1)
    gp = gp_fit(X_train, y_train)
    prob = gp.p_sample([0.2, 1.0, 1.0])
    print(prob)
    k = 100
    start = time.time()
    probs = gp.p_sample_batch(np.random.rand(k,3))
    end = time.time()
    print(probs)
    print(f'{k} samples done in {end-start:.2f} seconds time on {n} datapoints')
    return

if __name__ == '__main__':
    main()