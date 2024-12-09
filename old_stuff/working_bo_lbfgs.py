import scipy.optimize
import torch
import tensorflow as tf
import scipy
import utils
import numpy as np
import gpflow
import matplotlib.pyplot as plt

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

class NormalGp(DefaultMethodClass):
    def __init__(self, func_class : utils.test_functions.FuncToMinimise,
                 n_iters = 200,
                 n_random_evals = 20,
                 limits : list[tuple[float]] = [(-5.0,10.0),(0.0,15.0)]):
        
        super().__init__(func_class, limits)

        self.n_iters = n_iters - n_random_evals

        self.initial_random_evals(n_random_evals)

        self.iter_func_evals()

        self.method_name = "GP"

    
    def initial_random_evals(self, n_rand):
        samples = (np.random.rand(n_rand, self.dims) * self.limit_difs) + self.limit_mins
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
        acq = utils.acq_functions.EI_np(gp, self.limits)
        x = (np.random.rand(1, self.dims) * self.limit_difs) + self.limit_mins

        res = scipy.optimize.minimize(fun= acq.sample ,x0= x.flatten(), method='L-BFGS-B', bounds=self.limits)

        return res.x
            

    def iter_func_evals(self):
        for i in range(self.n_iters):
            if i%(self.n_iters//10) == 0:
                print(f'{i/(self.n_iters)*100} % done')
            sample = self.gen_sample()
            self.eval_sample(sample)
        print('Finished Evaling a GP')

class AverageMethod():
    def __init__(self, 
                 method_class : DefaultMethodClass,
                 func_class : utils.test_functions.FuncToMinimise,
                 n_method_instances : int = 30):
        self.method_objs = [method_class(func_class = func_class) for _ in range(n_method_instances)]
        self.sum_regret = []
        self.get_sum_regret()

        self.mean_regret = self.sum_regret / n_method_instances

    def get_sum_regret(self):
        self.sum_regret = self.method_objs[0].get_regret()
        for meth in self.method_objs[1:]:
            self.sum_regret += meth.get_regret()

    def get_result(self):
        return {'name': self.method_objs[0].method_name,
                'regret':self.mean_regret}
    
def plot_results(results):
    for result in results:
        plt.plot(result['regret'], label=result['name'])
        plt.legend()
    plt.yscale(value="log")
    plt.show()

def main():
    res_arr = []
    gp = AverageMethod(NormalGp, 
                  utils.test_functions.Branin,
                  n_method_instances=1)
    res_arr.append(gp.get_result())
    plot_results(res_arr)

if __name__ == '__main__':
    main()