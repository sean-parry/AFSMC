import gpflow
from gpflow.kernels import Matern52 as mt52
import numpy as np

import os, sys
sys.path.append(os.getcwd())
from SMC import prob_utils


class base_target():
    def p_sample(sample : list[float]):
        return
    
class normal_dist_2d():
    def p_sample(self, sample : list[float]):
        return prob_utils.multivariate_normal_p(sample, [3, 2], np.eye(2)*0.5)

class gp_fit():
    """
    sets aside data for a test train split for each p sample evaluates how
    well the params (sample) fits the data (something we want to maximise)
    """
    def __init__(self, X_train, y_train):
        
        return

    def p_sample(sample : list[float]):

        return


def main():

    return

if __name__ == '__main__':
    main()