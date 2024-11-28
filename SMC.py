import numpy as np
from Kernels import approx_opt
from Resamplers import Resampler
       

class SMC():
    def __init__(self, target_function, 
                 n_samples : int, 
                 n_iters : int, 
                 n_star : int = None,
                 kernel_class = approx_opt,
                 resample_alg : function = Resampler.stratified):
        self.target = target_function
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.n_star = self.n_samples/2 if n_star is None else n_star
        
        self.kernel_class = kernel_class
        self.resampler_alg = resample_alg
        self.Samples = None
        self.iter_main_loop()
    
    def norm_weights(self):
        self.weights /= sum(self.weights)

    def calc_neff(self):
        """could be slow may want ot change to the matrix way of
        doing this"""
        sum_w_squared = 0
        for w in self.weights:
            sum_w_squared +=  w**2
        return 1/sum_w_squared

    
    def new_proposal(self):
        new_proposal = []
        return new_proposal

    def proposal_probs(self, new_samples):
        """
        this will be based on the old proposal for random walk
        we will just need the covar matrix which should be a diag
        """
        return [1]*new_samples.shape()
    
    def new_weights(self, new_samples):
        new_weights = []
        l_kern_probs = self.kernel_class(self.samples,new_samples).probs
        proposal_probs = self.proposal_probs(new_samples)
        for i, new_s in enumerate(new_samples):
            expr1 = self.target(new_s)/self.target(self.samples[i])
            expr2 = l_kern_probs[i]/proposal_probs[i]
            new_w = self.weights[i] * expr1 * expr2
            new_weights.append(new_w)
        return new_weights

    def main_loop(self):
        self.norm_weights()
        Neff = self.calc_neff()
        if Neff< self.n_star:
            self.samples, self.weights = getattr(Resampler(self.samples,self.weights),self.resampler_alg)
        new_samples = self.new_proposal()
 
        self.weights = self.new_weights(new_samples)
        self.samples = new_samples

    def iter_main_loop(self, n_iters):
        for _ in range(self.n_iters):
            self.main_loop()


def main():
    smc = SMC()
    return

if __name__== '__main__':
    main()