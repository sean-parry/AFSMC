import numpy as np

class Approx_Opt():
    def __init__(self, old_samples, new_samples):
        self._x_arr = np.array(old_samples)
        self._x_hat_arr = np.array(new_samples)
        self.d = self._x_arr.shape[1]
        self.gen_gaussian()
        self.probs = []
        self.calc_probs()
        self.probs = np.array(self.probs)
        if max(self.probs >= 1):
            print(f'Warning: probability values greater than 1 inside'
                  f'of optimal l kernel, this may be due to singularities')

    def gen_gaussian(self):
        """samples should be a vstack of vectors
        if vector of samples is dimension m, and there
        are n samples then funciton expects a 
        """
        
        # generate the means and covars of all samples
        samples = np.hstack([self._x_arr]+[self._x_hat_arr])
        mu = np.mean(samples,axis=0)
        print(mu)
        # not sure abou tthe number 2 here
    
        self._mu_x, self._mu_xh = mu[:self.d], mu[self.d:]

        S = np.cov(samples.T)
        #S += np.eye(2*self.d)*1e-6 # avoid singularities

        self._Sxx = S[:self.d,:self.d]
        self._Sxhxh = S[self.d:,self.d:]
        self._Sxxh = S[:self.d,self.d:]
        self._Sxhxh_inv = np.linalg.inv(self._Sxhxh)
        self._Sxcxh = self._Sxx - self._Sxxh @ self._Sxhxh_inv @ self._Sxxh.T
        self._Sxcxh_inv = np.linalg.inv(self._Sxcxh)
        self.k = ((2*np.pi)**(-self.d/2))*((np.linalg.det(self._Sxcxh))**(-1/2))

    def bivar_normal_p(self, x, mu):
        x_minus_mu = x - mu
        expr1 = x_minus_mu.T @ self._Sxcxh_inv @ x_minus_mu
        return self.k*(np.exp(-0.5*expr1))
    
    def calc_probs(self):
        expr1 = self._Sxxh @ self._Sxhxh_inv
        for old_s, new_s in zip(self._x_arr, self._x_hat_arr):
            mu_xcxh = self._mu_x + expr1 @ (new_s-self._mu_xh)
            self.probs.append(self.bivar_normal_p(old_s, mu_xcxh))

def main():
    # fails for this test case gives values over 1
    n = 20
    x1 = np.random.rand(n,2)
    x2 = np.random.rand(n,2)
    print(Approx_Opt(x1,x2).probs[:10])
    return

if __name__ == '__main__':
    main()
