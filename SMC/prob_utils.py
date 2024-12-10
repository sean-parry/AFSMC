import numpy as np
from scipy.stats import multivariate_normal

def multivariate_normal_p(sample : list[float],
                          mean : list[float], 
                          covar_mat : list[list[float]]):
    
    return multivariate_normal.pdf(sample, mean=mean, cov=covar_mat)