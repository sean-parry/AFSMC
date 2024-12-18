import matplotlib.pyplot as plt
import numpy as np

def plot_results(results : dict):
    for result in results:
        plt.plot(result['regret'], label=result['name'])
        plt.legend()
    plt.yscale(value="log")
    plt.show()

class plot_ei_2d():
    """
    really just a function to plot the ei of a gp or wieghted
    average of gps over a branin function, but will technically
    do any 2d problem.

    makes an np.linspace within the limits for a given step size
    samples the acquisition function for all of these points and
    then plots this on a graph, (tbf might be best to just return
    the fit, axes instead)

    samples_acq_func : funciton
    a funciuton that should take a position value and return the 
    expected improvment
    """
    def __init__(self, 
                 sample_acq_func,
                 limits: list[tuple[float]] = [(-5.0, 10.0), (0.0, 15.0)],
                 step_size: float = 0.25):
        
        self.limits = limits
        self.targ = sample_acq_func
        self.step_size = step_size
        self._plot_results()

    def _generate_grid(self):
        x = np.arange(self.limits[0][0], self.limits[0][1], self.step_size)
        y = np.arange(self.limits[1][0], self.limits[1][1], self.step_size)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def _evaluate_function(self):
        # this only runs on one core so is painfully slow on laptop
        # not sure what the best way to distribute the task would
        # be, also a status indicator would be good to print to 
        # terminal
        X, Y = self._generate_grid()
        Z = np.vectorize(lambda x, y: self.targ([x, y]))(X, Y)
        return X, Y, Z

    def _plot_results(self):
        X, Y, Z = self._evaluate_function()
        print('done the computation')
        plt.figure(figsize=(10, 6))
        heatmap = plt.pcolormesh(X,Y,Z, shading='auto', cmap='viridis')
        plt.colorbar(heatmap, label="Target Function Output")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Target Function Heatmap")
        plt.show()

def main():
    def test_func(sample):
        x, y = sample
        return np.random.rand(1)[0]
    plot_ei_2d(sample_acq_func=test_func)
    return

if __name__ == '__main__':
    main()