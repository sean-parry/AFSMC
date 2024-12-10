import utils
import SMC


def main():
    res = []
    avg_gp = utils.average_method.AverageMethod(
        method_class = utils.methods.NormalGp,
        func_class = utils.test_functions.Branin,
        n_method_instances = 1
    )
    res.append(avg_gp.get_result())
    utils.plotter.plot_results(res)

if __name__ == '__main__':
    main()