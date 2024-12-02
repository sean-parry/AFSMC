import numpy as np
from scipy.stats import multivariate_normal


"""
For approx opt its worth mentioning that the we are doing unnecessary matrix inversions
that could be avoided if we didn't use multivar_normal from scipy stats and just wrote the eq ourselves
although doing that causes an error getting vals above 1 = which tbf isn't actually resolved by using
scipy stats anyway - could possibly normalise the result - also i guess it shouldn't matter as after running the kernel
the weights are normalised so even if the p values were greater than one as long as they are correct with
respect to eachother then it shouldn't matter
"""
class Approx_Opt():
    def __init__(self, old_samples, new_samples):
        self._x_arr = np.array(old_samples)
        self._x_hat_arr = np.array(new_samples)
        self.d = self._x_arr.shape[1]
        self._gen_gaussian()
        self.probs = []
        self._calc_probs()
        self.probs = np.array(self.probs)
        return
        # its actually ok for vals to be above 1 bc we have a prob density 
        # function that just need to integrate to one
        if max(self.probs >= 1):
            print(f'Warning: probability values greater than 1 inside of optimal l kernel, this may be due to singularities')
            pass

    def _gen_gaussian(self):
        """samples should be a vstack of vectors
        if vector of samples is dimension m, and there
        are n samples then funciton expects a 
        """
        
        # generate the means and covars of all samples
        samples = np.hstack([self._x_arr]+[self._x_hat_arr])
        mu = np.mean(samples,axis=0)
        # not sure abou tthe number 2 here
    
        self._mu_x, self._mu_xh = mu[:self.d], mu[self.d:]

        S = np.cov(samples.T)
        S += np.eye(2*self.d)*1e-6 # avoid singularities

        self._Sxx = S[:self.d,:self.d]
        self._Sxhxh = S[self.d:,self.d:]
        self._Sxxh = S[:self.d,self.d:]
        self._Sxhxh_inv = np.linalg.inv(self._Sxhxh + np.eye(self.d) * 1e-6)
        self._Sxcxh = self._Sxx - self._Sxxh @ self._Sxhxh_inv @ self._Sxxh.T
        """
        bascially i can't use my own bivar normal calc bc it is wrong in some way
        for small covar matricies so going to use scipy instead - since the dims
        of the hyperparam space never get too high this shouldn't be too slow 
        """
        self._Sxcxh_inv = np.linalg.inv(self._Sxcxh)
        self.k = ((2*np.pi)**(-self.d/2))*((np.linalg.det(self._Sxcxh))**(-1/2))

    def _bivar_normal_p(self, x, mu):
        x_minus_mu = x - mu
        expr1 = x_minus_mu.T @ self._Sxcxh_inv @ x_minus_mu
        return self.k*(np.exp(-0.5*expr1))
        # log_p = multivariate_normal.logpdf(x,mu, self._Sxcxh)
        # p = np.exp(log_p)
        # return p
    
    def _calc_probs(self):
        expr1 = self._Sxxh @ self._Sxhxh_inv
        for old_s, new_s in zip(self._x_arr, self._x_hat_arr):
            mu_xcxh = self._mu_x + expr1 @ (new_s-self._mu_xh)
            p = self._bivar_normal_p(old_s, mu_xcxh)
            if np.isnan(p):
                p = 0
            self.probs.append(p)

def main():
    # fails for this test case gives values over 1
    np.random.seed(42)
    n = 20
    x1 = np.random.rand(n,2)
    x2 = np.random.rand(n,2)
    print(Approx_Opt(x1,x2).probs[:10])
    return

if __name__ == '__main__':
    main()
