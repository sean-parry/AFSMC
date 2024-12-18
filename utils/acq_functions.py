# should put my expected imporvement function here
import gpflow
import torch
import numpy as np

class EI_np():
    def __init__(self, gp: gpflow.models.GPR, limits: list[tuple]):
        self.gp = gp
        self.limits = limits

    def _expected_improvement(self, x, y_best):
        mean_sample, var_sample = self.gp.predict_f(x)
        mean_sample = torch.tensor(mean_sample.numpy(), dtype=torch.float64)
        var_sample = torch.tensor(var_sample.numpy(), dtype=torch.float64)

        std = torch.sqrt(var_sample)
        std = torch.maximum(std, torch.tensor(1e-9, dtype=torch.float64))

        z = (y_best-mean_sample) / std

        ei = (y_best-mean_sample) * torch.sigmoid(z) + std * torch.exp(-0.5 * torch.square(z)) / torch.tensor(np.sqrt(2 * np.pi), dtype=torch.float64)

        return ei

    def sample(self, x : list[float]):
        x = np.array(x).reshape(1,-1)
        y_best_np = np.min(self.gp.data[1]) 
        y_best = torch.tensor(y_best_np, dtype=torch.float64)

        ei = torch.tensor([x.shape[0]], dtype=torch.float64)

        ei = self._expected_improvement(x, y_best)
        
        return ei.squeeze().numpy()

class EI_torch:
    def __init__(self, gp: gpflow.models.GPR, limits: list[tuple]):
        self.gp = gp
        self.limits = limits

    def _expected_improvement(self, x, y_best):
        mean, var = self.gp.predict_f(x.detach().numpy().reshape((1, -1)))
        mean = torch.tensor(mean.numpy(), dtype=torch.float64)
        var = torch.tensor(var.numpy(), dtype=torch.float64)

        std = torch.sqrt(var)
        std = torch.maximum(std, torch.tensor(1e-9, dtype=torch.float64))

        z = (mean - y_best) / std

        ei = (mean - y_best) * torch.sigmoid(z) + std * torch.exp(-0.5 * torch.square(z)) / torch.tensor(np.sqrt(2 * np.pi), dtype=torch.float64)

        return ei

    def sample(self, x):

        y_best_np = np.min(self.gp.data[1]) 
        y_best = torch.tensor(y_best_np, dtype=torch.float64)

        ei = torch.tensor([x.shape[0]], dtype=torch.float64)

        ei = self._expected_improvement(x, y_best)

        return ei

class test():
    def __init__(self):
        self.limits = [(-5.0,10.0),(0.0,15.0)]
        self.limit_difs = [abs(l1-l2) for l1,l2 in self.limits]
        self.limit_mins = [min(l1,l2) for l1, l2 in self.limits]
        self.dims = len(self.limits)
        import test_functions # for testing remove later

        self.n_random_evals = 20
        self.br = test_functions.Branin()
        self.X_train, self.y_train = [], []

    def eval_sample(self, sample):
        try:
            self.X_train = np.vstack((self.X_train, sample))
            self.y_train = np.vstack((self.y_train , self.br.eval(sample)))
        except:
            self.X_train = sample
            self.y_train = self.br.eval(sample)
    
    def initial_samples(self):
        samples = (np.random.rand(self.n_random_evals, self.dims) * self.limit_difs) + self.limit_mins
        for sample in samples:
            self.eval_sample(sample)
    
    def get_gp(self):
        model = gpflow.models.GPR(
            (self.X_train, self.y_train),
            kernel = gpflow.kernels.Matern52())
        
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        return model
    
    def gen_sample(self):
        self.initial_samples()
        gp = self.get_gp()
        x = [0.0, 0.0]

        acq = EI_np(gp, self.limits)
        
        print(acq.sample(x))

        
def test_func():
    t = test()
    t.gen_sample()

if __name__ =='__main__':
    test_func()
