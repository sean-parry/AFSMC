import numpy as np
from Kernels import Approx_Opt
import Resamplers
import Proposals
import Initial_Proposals
import Prob_Utils
       

class SMC():
    def __init__(self, target_function, 
                 n_samples : int, 
                 n_iters : int, 
                 initial_proposal_obj = Initial_Proposals.Initial_Proposal(),
                 proposal_obj = Proposals.Proposal(),
                 kernel_class = Approx_Opt,
                 resample_obj = Resamplers.Staratified(),
                 n_star : int = None):
        self.target = target_function
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.n_star = self.n_samples/2 if n_star is None else n_star
        
        self.initial_proposal_obj = initial_proposal_obj
        self.proposal_obj = proposal_obj
        self.kernel_class = kernel_class
        self.resample_obj = resample_obj
        self.Samples = None
        self.iter_main_loop()
    
    def norm_weights(self):
        sum_weights = sum(self.weights)
        if sum_weights == 0 or np.isnan(sum_weights):
            self.weigths = [1,self.n_samples]*self.n_samples
        else:
            self.weights /= sum_weights

    def calc_neff(self):
        """could be slow may want ot change to the matrix way of
        doing this"""
        sum_w_squared = 0
        for w in self.weights:
            sum_w_squared +=  w**2
        return 1/sum_w_squared

    def new_proposal(self):
        self.proposal_obj.update(self.samples)
        return self.proposal_obj.new_samples
    
    def new_weights(self, new_samples):
        new_weights = []
        l_kern_probs = self.kernel_class(self.samples,new_samples).probs
        proposal_probs = self.proposal_obj.probs
        # print(self.weights)
        for i, new_s in enumerate(new_samples):
            expr1 = self.target(new_s)/self.target(self.samples[i])
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
        for p, s in zip(probs, self.samples):
            initial_weights.append(self.target(s)/p)
        self.weights = np.array(initial_weights)


    def iter_main_loop(self):
        self.initialise_samples_and_weights()
        for _ in range(self.n_iters):
            self.main_loop()
        self.norm_weights()


def main():
    def target_function(sample):
        """
        takes a vector sample returns a probability
        """
        return Prob_Utils.multivariate_normal_p(sample, [3, 2], np.eye(2))
    
    smc = SMC(target_function=target_function,
              n_samples=100,
              n_iters=200,
              initial_proposal_obj=Initial_Proposals.Strandard_Gauss_Noise(dim_samples = 2, n_samples=100),
              proposal_obj= Proposals.Random_Walk(step_sizes = [0.2,0.2]))
    
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