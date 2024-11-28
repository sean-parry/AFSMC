import numpy as np

def multivariate_normal_p(sample : list[float],
                          mean : list[float], 
                          covar_mat : list[list[float]]):
    
    x_minus_mean = np.subtract(sample,mean)
    temp1 = np.dot(x_minus_mean.T,np.linalg.inv(covar_mat))
    temp2 = np.dot(temp1,x_minus_mean)
    n = len(sample)
    k = (((2*np.pi)**(n/2))*((np.linalg.det(covar_mat))**(1/2)))**-1
    return k*(np.exp(-0.5*temp2))