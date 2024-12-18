from skopt.benchmarks import branin as _branin

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
    
    def _update_regret(self):
        if self.regret_arr:
            regret = min(self.regret_arr[-1], abs(self.optimal_val-self._eval_vals[-1]))
        else:
            regret = abs(self.optimal_val-self._eval_vals[-1])
        self.regret_arr.append(regret)

    def eval(self, x):
        self.num_evals += 1
        self._eval_points.append(x)
        self._eval_vals.append(_branin(x))
        self._update_regret()
        return self._eval_vals[-1]

def main():
    import plotter
    plotter.plot_ei_2d(Branin().eval)

if __name__ == '__main__':
    main()