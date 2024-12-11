import gpflow
from gpflow.kernels import Matern52 as mt52
import numpy as np
import tensorflow as tf

import os, sys
sys.path.append(os.getcwd())
from SMC import prob_utils


class base_target():
    def p_sample(sample : list[float])->float:
        return
    def p_sample_batch(sample : list[list[float]])->list[float]:
        return
    
class normal_dist_2d(base_target):
    def p_sample(self, sample : list[float]):
        return prob_utils.multivariate_normal_p(sample, [3, 2], np.eye(2)*0.5)

class gp_fit(base_target):
    """
    sets aside data for a test train split for each p sample evaluates how
    well the params (sample) fits the data (something we want to maximise)
    not sure if this is the best way or even the way that gpflow does it in
    their optimizers
    """
    def __init__(self, X_all, y_all, train_test_split = 0.8):
        self.X_all = np.array(X_all)
        self.y_all = np.array(y_all)
        self.n_samples = self.y_all.shape[0]

        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_splitter(train_test_split)
        
    
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

    def p_sample(self, sample : list[float]):
        """
        the sample is the hyperparams for the gp over the model space
        """
        variance = sample[:1][0]
        lengths = (sample[1:])
        kernel = mt52(variance=variance, lengthscales=lengths)
        model = gpflow.models.GPR((self.X_train, self.y_train),
                          kernel=kernel)
        

        mean, cov = model.predict_y(self.X_test, full_cov=False)
        mean = np.squeeze(mean.numpy())
        cov = np.squeeze(cov.numpy())

        return prob_utils.multivariate_normal_p(mean, np.squeeze(self.y_test), np.diag(cov))
    
    def p_sample_batch(self, samples : list[list[float]]):
        probs = []
        for sample in samples:
            variance = sample[:1][0]
            lengths = (sample[1:])
            kernel = mt52(variance=variance, lengthscales=lengths)
            model = gpflow.models.GPR((self.X_train, self.y_train),
                            kernel=kernel)
        
            mean, cov = model.predict_y(self.X_test, full_cov=False)
            mean = np.squeeze(mean.numpy())
            cov = np.squeeze(cov.numpy())
            probs.append(prob_utils.multivariate_normal_p(mean, np.squeeze(self.y_test), np.diag(cov)))

        return np.array(probs)


def main():
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