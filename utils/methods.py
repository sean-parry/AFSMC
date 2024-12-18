import gpflow
import numpy as np
import scipy

import os, sys
sys.path.append(os.getcwd())
from utils import acq_functions, test_functions, plotter
import SMC

class DefaultMethodClass():
    def __init__(self, func_class,
                 limits
                 ):
        self.method_name = 'Default Method Class'


        self.X_train = []
        self.y_train = []

        self.func_obj = func_class()

        self.limits = limits
        self.dims = len(limits)
        self.limit_difs = [abs(l1-l2) for l1,l2 in limits]
        self.limit_mins = [min(l1,l2) for l1, l2 in limits]

        return
    
    def get_regret(self):
        if self.func_obj.regret_arr:
            return np.array(self.func_obj.regret_arr)
        else:
            print('Funciton to minimise not given or Search method')

    def eval_sample(self, sample):
        try:
            self.X_train = np.vstack((self.X_train, sample))
            self.y_train = np.vstack((self.y_train , self.func_obj.eval(sample)))
        except:
            self.X_train = sample
            self.y_train = self.func_obj.eval(sample)
    
    def run(self):
        return


class NormalGp(DefaultMethodClass):
    def __init__(self, func_class : test_functions.FuncToMinimise,
                 n_iters = 200,
                 n_random_evals = 20,
                 limits : list[tuple[float]] = [(-5.0,10.0),(0.0,15.0)],
                 random_eval_seed : int = None):
        
        super().__init__(func_class, limits)

        self.n_iters = n_iters - n_random_evals
        self.n_random_evals = n_random_evals
        self.random_eval_seed = random_eval_seed

        self.method_name = "GP"

    def _initial_random_evals(self):
        np.random.seed(seed=self.random_eval_seed)
        samples = (np.random.rand(self.n_random_evals, self.dims) * self.limit_difs) + self.limit_mins
        for x in samples:
            self.eval_sample(x)
    
    def get_gp(self):
        model = gpflow.models.GPR(
            (self.X_train, self.y_train),
            kernel = gpflow.kernels.Matern52())
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        return model

    def gen_sample(self):
        gp = self.get_gp()
        acq = acq_functions.EI_np(gp, self.limits)
        # might want this to start at curr best x idk tho
        x = (np.random.rand(1, self.dims) * self.limit_difs) + self.limit_mins

        res = scipy.optimize.minimize(fun= -1*acq.sample ,x0= x.flatten(), method='L-BFGS-B', bounds=self.limits)
        return res.x
            
    def _iter_func_evals(self):
        for i in range(self.n_iters):
            if i%(self.n_iters//10) == 0:
                print(f'{i/(self.n_iters)*100} % done')
            sample = self.gen_sample()
            print(sample)
            self.eval_sample(sample)
        print('Finished Evaling a GP')
    
    def run(self):
        self._initial_random_evals()
        self._iter_func_evals()

    def test_plotter(self):
        self._initial_random_evals()
        gp = self.get_gp()
        acq = acq_functions.EI_np(gp, self.limits)
        print('got to the plotting bit')
        plotter.plot_ei_2d(sample_acq_func=acq.sample)
        return
        


class SMC_GP(NormalGp):
    """
    probably should just move some stuff around and not inherit
    from normal gp and instead inherit from default method class
    since i overwrite 90% of stuff anyway but then default method
    class would need more vars
    """
    def __init__(self, func_class : test_functions.FuncToMinimise,
                 smc_obj = SMC.smc_search.SMC(),
                 n_iters = 200,
                 n_random_evals = 20,
                 limits : list[tuple[float]] = [(-5.0,10.0),(0.0,15.0)],
                 random_eval_seed : int = None):
        super().__init__(func_class=func_class,
                         n_iters=n_iters,
                         n_random_evals=n_random_evals,
                         limits=limits,
                         random_eval_seed=random_eval_seed)
        self.smc_obj = smc_obj
        self.method_name = 'SMC GP'

    def run_smc(self)->list[np.ndarray, np.ndarray]:
        # update X_train and y_train in the smc obj and run it
        self.smc_obj.update_target_vars(self.X_train, self.y_train)
        weights, samples = self.smc_obj.run()
        """weights = [0.1,0.3,0.6]
        samples = [[1,2,3],
                   [1,1,1],
                   [1,3,3]]"""
        return weights, samples

    def average_acq_fun(self):
        models = []
        self.weights, samples = self.run_smc()
        for s in samples:
            models.append(gpflow.models.GPR(
            (self.X_train, self.y_train),
            kernel = gpflow.kernels.Matern52(
                variance=s[0],
                lengthscales=s[1:])))
        self.acq_funcs = [acq_functions.EI_np(gp, self.limits) for gp in models]
        return
    
    def sample_average_acq(self, x):
        ans = []
        for acq_func in self.acq_funcs:
            ans.append(acq_func.sample(x))
        ans = np.array(ans)
        return sum(ans*self.weights)
    
    def gen_sample(self):
        self.average_acq_fun()
        self.weights, self.samples = self.run_smc()
        x = (np.random.rand(1, self.dims) * self.limit_difs) + self.limit_mins
        res = scipy.optimize.minimise(fun= -1*self.sample_average_acq ,x0= x.flatten(), method='L-BFGS-B', bounds=self.limits)
        return res.x


def main():
    # maybe a sample size incease inside of the smc would reduce
    # the weight degredation especially since you'd get more
    # so better intial proposals (after a resample) arguably
    # you should start with a large proposal and then cut down the
    # samples to something more manageable - but arguably the divergence
    # comes from the walking anyway so this wouldn't really make a large
    # difference in the accuracy of a gp especially one focusing on a
    # static target
    normalgp = NormalGp(func_class=test_functions.Branin,
                        n_random_evals=200,
                        random_eval_seed=None)
    normalgp.test_plotter()
    return
    smc_gp = SMC_GP(
        func_class = test_functions.Branin,
        smc_obj=SMC.smc_search.SMC(
            target_obj=SMC.target_functions.gp_fit(),
            n_samples = 30,
            n_iters = 20,
            initial_proposal_obj=SMC.initial_proposals.Gauss(),
            proposal_obj=SMC.proposals.Defensive_Sampling(),
    ),
    n_iters=30)

    smc_gp.run()
    print(smc_gp.get_regret())
    return

if __name__ == '__main__':
    main()

"""
just saving an output of a 20 random evals, 10 iters of smc
search (at 30 samples, 20 iters)  remove once better proposal 
is written:
not sure why it evals the same point twice to begin with that could
point to some underlying issue that need solving

[10.  0.]
10.0 % done
[10.  0.]
20.0 % done
[2.90469878 0.78380738]
30.0 % done
[9.12073278 2.24607226]
40.0 % done
[9.30237569 0.77968554]
50.0 % done
[-2.50583354 15.        ]
60.0 % done
[-0.83464552  2.58877061]
70.0 % done
[10.         1.6211035]
80.0 % done
[6.05687869 1.20225637]
90.0 % done
[3.58288607 1.40066145]
Finished Evaling a GP
[69.44142142 12.79568249 12.79568249  6.7450766   6.7450766   1.39248889
  1.39248889  1.39248889  1.39248889  1.39248889  1.39248889  1.39248889
  1.39248889  1.39248889  1.39248889  1.39248889  1.39248889  1.39248889
  1.39248889  1.39248889  1.39248889  1.39248889  1.39248889  0.44066144
  0.44066144  0.44066144  0.44066144  0.44066144  0.44066144  0.44066144]
"""