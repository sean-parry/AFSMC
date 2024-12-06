from skopt.benchmarks import branin as _branin
import torch
import math

class FuncToMinimise():
    def __init__(self):
        self.regret_arr = []
        self.num_evals = 0
    
    def eval(self, x):
        print('This is a base class please specify a child')
        return

class Branin(FuncToMinimise):
    def __init__(self):
        super().__init__()
        self.optimal_val = 0.397887
        self._eval_points = []
        self._eval_vals = []
    
    def _update_regret(self, result):
        result = result.item()
        self._eval_vals.append(result)
        print(result)
        try:
            regret = min(self.regret_arr[-1], abs(self.optimal_val-self._eval_vals[-1]))
        except:
            regret = abs(self.optimal_val-self._eval_vals[-1])
        self.regret_arr.append(regret)

    def eval(self, X):
        self.num_evals += 1

        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        t1 = X[:, 1] - 5.1 / (4 * math.pi ** 2) * X[:, 0] ** 2 + 5 / math.pi * X[:, 0] - 6
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[:, 0])
        B = t1 ** 2 + t2 + 10

        self._update_regret(B)

        result = B

        return result if batch else result.squeeze(0)



def main():
    br = Branin()
    x = torch.tensor([0.0,0.0], requires_grad=True)
    print(x)
    print(br.eval(x))


if __name__ == '__main__':
    main()