import matplotlib.pyplot as plt

def plot_results(results):
    for result in results:
        plt.plot(result['regret'], label=result['name'])
        plt.legend()
    plt.yscale(value="log")
    plt.plot()