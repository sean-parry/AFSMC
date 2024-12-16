import numpy as np

#probably not the pretiest way of fixing the import issues but it works
import os, sys
sys.path.append(os.getcwd())
from SMC import prob_utils, initial_proposals, kernels, proposals, resamplers, target_functions

class SMC():
    def __init__(self, target_obj = target_functions.base_target(), 
                 n_samples : int = 100, 
                 n_iters : int = None,
                 initial_proposal_obj = initial_proposals.Initial_Proposal(),
                 proposal_obj = proposals.Proposal(),
                 kernel_class = kernels.Approx_Opt,
                 resample_obj = resamplers.Staratified(),
                 n_star : int = None):
        self.target_obj = target_obj
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.n_star = self.n_samples/2 if n_star is None else n_star
        
        initial_proposal_obj.n_samples = n_samples
        self.initial_proposal_obj = initial_proposal_obj
        self.proposal_obj = proposal_obj
        self.kernel_class = kernel_class
        self.resample_obj = resample_obj
        self.samples = None
        #self.run()
    
    def norm_weights(self):
        sum_weights = sum(self.weights)
        if sum_weights == 0 or np.isnan(sum_weights):
            self.weigths = [1,self.n_samples]*self.n_samples
        else:
            new_weights = [w/sum_weights for w in self.weights]
            self.weights = new_weights

    def calc_neff(self):
        """could be slow may want ot change to the matrix way of
        doing this"""
        sum_w_squared = 0
        for w in self.weights:
            print(self.weights)
            sum_w_squared +=  w**2
        return 1/sum_w_squared

    def new_proposal(self):
        self.proposal_obj.update(self.samples)
        return self.proposal_obj.new_samples
    
    def new_weights(self, new_samples):
        new_weights = []
        l_kern_probs = self.kernel_class(self.samples,new_samples).probs
        proposal_probs = self.proposal_obj.probs
        s_target_probs = self.target_obj.p_sample_batch(self.samples)
        ns_target_probs = self.target_obj.p_sample_batch(new_samples)
        for i, _ in enumerate(new_samples):
            expr1 = ns_target_probs[i]/s_target_probs[i]
            expr2 = l_kern_probs[i]/proposal_probs[i]
            # expr2 = 1
            new_w = self.weights[i] * expr1 * expr2
            if np.isnan(new_w):
                new_w = 1e-6
            new_weights.append(new_w)
        return new_weights

    def main_loop(self):
        self.norm_weights()
        Neff = self.calc_neff()
        #print(self.weights)
        if Neff< self.n_star:
            self.samples, self.weights = self.resample_obj.resample(self.samples, self.weights)
        new_samples = self.new_proposal()
 
        self.weights = self.new_weights(new_samples)
        self.samples = new_samples

    def initialise_samples_and_weights(self):
        probs, self.samples = self.initial_proposal_obj.sample()
        initial_weights = []
        targ_probs = self.target_obj.p_sample_batch(self.samples)
        for p, targ_p in zip(probs, targ_probs):
            initial_weights.append(targ_p/p)
        self.weights = np.array(initial_weights)


    def run(self)->tuple[np.ndarray, np.ndarray]:
        """
        returns weights, samples
        every time run is called we run the smc again - nothing is kept from 
        the previous run, if you wish to change something about the smc then
        just change one of the smc objects variables then call run
        """
        self.initialise_samples_and_weights()
        for _ in range(self.n_iters):
            self.target_obj
            self.main_loop()
        self.norm_weights()
        return self.weights, self.samples


def main():
    smc = SMC(target_obj=target_functions.normal_dist_2d(),
              n_samples=100,
              n_iters=100,
              initial_proposal_obj=initial_proposals.Strandard_Gauss_Noise(dim_samples = 2),
              proposal_obj= proposals.Defensive_Sampling())
    
    for _ in range(2):
        smc.run()
        sum_w = 0
        ans = [0,0]
        for s, w in zip(smc.samples, smc.weights):
            # print(f'sample value: {s}, weight value {w}')
            sum_w += w
            ans += s*w
        print(f'total weight sum: {sum_w}')
        print(f'estimate is {ans}')
    return

if __name__== '__main__':
    main()

"""
Its hard to say whether we actually see an improvement using the approx opt l kernel
just choosing the l kernel to cancel with the forward proposal - also worth saying that 
using random walk without varying the step size will just make the output very poor

should just try a more sensible proposal - are one with gradients acceptable or does 
that not allow a weight update, 
"""