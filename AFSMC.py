import SMC
import numpy as np
import gpflow
import Initial_Proposals
import Proposals
"""
The goal of the file is to for a set of data make a gp with each sample output
of the smc which is trying to pick samples to minimise LOOCV or kl divergence
then find the minimisation of the global acquisition funciton
"""

class AFSMC():
    def __init__(self, dataset, 
                 smc_obj : SMC.SMC, 
                 acq_func : str, 
                 gp_kernel : gpflow.kernels = gpflow.kernels.Matern52):
        self.dataset = dataset
        self.smc_obj = smc_obj
        self.acq_func = acq_func
        self.gp_kernel = gp_kernel

        self.gp_array = []
        self.acq_func_array = [] # minimise the additive weight*acqfunc meta function
        next_eval_point = 1
        return next_eval_point = 1


    def _populate_gp_array(self):

    
    def get_next_eval_point(self):
        """
        causes the run of the smc gets the vals from the smc, does the acq func 
        stuff ect
        """
        # makes the smc target with the current dataset
        # smc.run()

    def update_dataset(self):
        """
        updates the dataset with the new point that has been evaluated
        """

def main():
    smc_obj = SMC.SMC(
        target_function= None,
        n_samples= 50,
        n_iters= 10,
        initial_proposal_obj = Initial_Proposals.Strandard_Gauss_Noise(dim_samples=2),
        proposal_obj = Proposals.Random_Walk()

    )
    AFSMC
    return


if __name__ == '__main__':
    main()